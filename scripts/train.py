import argparse
import os
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from gpt.data.dataloader import FineWebEduDataModule
from gpt.modeling.callbacks import (
    HellaSwagEvalutor,
    TextGenerationCallback,
    TokensPerSecond,
)
from gpt.modeling.module import GPTModule
from gpt.modeling.schemas import GenerationConfig, GPTConfig, OptimizerConfig

PROJECT_NAME = "gpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--hellaswag_data_path",
        type=str,
        required=True,
        help="Path to the hellaswag data directory",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Path to the results directory"
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    # set the default floating point precision
    torch.set_float32_matmul_precision("high")

    # set random seed
    L.seed_everything(1337)

    # Define the model configuration
    config = GPTConfig(vocab_size=50304)
    # config = GPTConfig()

    # set batch size and gradient accumulation steps
    total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
    batch_size = 16
    gradient_accumulation_steps = total_batch_size // (batch_size * config.block_size)
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    # Define the optimizer configuration
    optimizer_config = OptimizerConfig(
        warmup_steps=715 * gradient_accumulation_steps,
    )

    # Define the data module
    # data_module = ShakespeareDataModule(
    #     data_path=args.data_path,
    #     block_size=config.block_size,
    #     batch_size=batch_size,
    # )
    data_module = FineWebEduDataModule(
        data_path=args.data_path,
        block_size=config.block_size,
        batch_size=batch_size,
        num_workers=4,
    )

    # Define the model
    model = GPTModule(config, optimizer_config)

    # Define the logger
    experiment_name = f"{PROJECT_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    experiment_dir = os.path.join(args.result_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    logger = WandbLogger(
        name=experiment_name,
        project=PROJECT_NAME,
        save_dir=experiment_dir,
    )

    # define callbacks
    tokens_per_second = TokensPerSecond()
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        dirpath=experiment_dir + "/checkpoints",
    )
    hellaswag = HellaSwagEvalutor(
        data_dir=args.hellaswag_data_path,
        split="val",
    )
    text_generation = TextGenerationCallback(
        generation_config=GenerationConfig(),
        prompt="Hello, I'm a language model,",
        num_return_sequences=5,
    )
    callbacks = [
        tokens_per_second,
        lr_monitor,
        model_checkpoint,
        hellaswag,
        text_generation,
    ]

    # Define the trainer
    trainer = L.Trainer(
        # fast_dev_run=True,
        logger=logger,
        max_epochs=5,
        precision="bf16-mixed",
        accumulate_grad_batches=gradient_accumulation_steps,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        # log_every_n_steps=1,
        enable_checkpointing=True,
        # max_steps=100,
        val_check_interval=250 * gradient_accumulation_steps,
        # limit_val_batches=50,
        # check_val_every_n_epoch=2,
        # accelerator="gpu",
    )

    # Train the model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
