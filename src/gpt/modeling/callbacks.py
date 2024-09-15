import json
import os
import time
from typing import Any, Dict

import lightning as L
import tiktoken
import torch
from tqdm import tqdm

from gpt.modeling.schemas import GenerationConfig
from gpt.utils.hellaswag import download_file, get_most_likely_row


class TokensPerSecond(L.Callback):
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.start_time = None

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.total_tokens = 0
        self.start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.total_tokens += batch[0].numel()
        elapsed_time = time.time() - self.start_time
        tokens_per_sec = self.total_tokens / elapsed_time
        trainer.logger.log_metrics(
            {"tokens_per_sec": tokens_per_sec}, step=trainer.global_step
        )


class HellaSwagEvalutor(L.Callback):
    hellaswags = {
        "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
        "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
        "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
    }

    def __init__(
        self,
        split: str,
        data_dir: str,
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.enc = tiktoken.get_encoding("gpt2")

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # prepare the data for evaluation
        os.makedirs(self.data_dir, exist_ok=True)
        data_url = self.hellaswags[self.split]
        data_path = os.path.join(self.data_dir, f"hellaswag_{self.split}.jsonl")
        if not os.path.exists(data_path):
            download_file(data_url, data_path)
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # evaluate on the held-out set
        correct = 0
        total = 0
        for example in tqdm(self.data, desc=f"{self.split} eval"):
            _, tokens, mask, label = self.render_example(example)
            # move the data to the device
            tokens = tokens.to(pl_module.device)
            mask = mask.to(pl_module.device)
            with torch.no_grad():
                logits = pl_module.model(tokens)
            pred = get_most_likely_row(tokens, mask, logits)
            if pred == label:
                correct += 1
            total += 1
        acc = correct / total
        trainer.logger.log_metrics({f"{self.split}_acc": acc}, step=trainer.global_step)

    def render_example(self, example: Dict[str, Any]) -> str:
        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]

        # data needed to reproduce this eval on the C size
        data = {
            "label": label,
            "ctx_tokens": None,
            "ending_tokens": [],
        }

        # gather up all the tokens
        ctx_tokens = self.enc.encode(ctx)
        data["ctx_tokens"] = ctx_tokens
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = self.enc.encode(
                " " + end
            )  # note: prepending " " because GPT-2 tokenizer
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
            data["ending_tokens"].append(end_tokens)

        # have to be careful during the collation because the number of tokens
        #  in each row can differ
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            mask[i, : len(mask_row)] = torch.tensor(mask_row)

        return data, tokens, mask, label


class TextGenerationCallback(L.Callback):
    def __init__(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        num_return_sequences: int = 1,
    ):
        super().__init__()
        self.prompt = prompt
        self.generation_config = generation_config
        self.num_return_sequences = num_return_sequences
        self.enc = tiktoken.get_encoding("gpt2")

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.prompt_tokens = self.enc.encode(self.prompt)
        self.prompt_tokens = torch.tensor(self.prompt_tokens, dtype=torch.long)
        self.prompt_tokens = self.prompt_tokens.unsqueeze(0).repeat(
            self.num_return_sequences, 1
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # generate text from the prompt
        sample_rng = torch.Generator(device=pl_module.device)
        # ste seed for reproducibility using the rank of the process
        sample_rng.manual_seed(42 + trainer.global_rank)
        with torch.no_grad():
            generated = pl_module.model.generate(
                self.generation_config, self.prompt_tokens, device_type=pl_module.device
            )
        data = []
        for i in range(self.num_return_sequences):
            generated_text = self.enc.decode(generated[i].cpu().numpy())
            data.append([self.prompt, generated_text])

        trainer.logger.log_text(
            key="generated_text",
            columns=["prompt", "generated"],
            data=data,
            step=trainer.global_step,
        )
