import multiprocessing as mp
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


class FineWebEdu:
    def __init__(self, local_dir, remote_name, shard_size):
        self.local_dir = local_dir
        self.remote_name = remote_name
        self.shard_size = shard_size
        os.makedirs(self.local_dir, exist_ok=True)
        self.fw = load_dataset(
            "HuggingFaceFW/fineweb-edu", name=remote_name, split="train"
        )
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]

    def tokenize(self, doc):
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (
            tokens_np < 2**16
        ).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def write_datafile(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def write_shards(self):
        nprocs = max(1, os.cpu_count() // 2)
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            for tokens in pool.imap(self.tokenize, self.fw, chunksize=16):
                if token_count + len(tokens) < self.shard_size:
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(
                            total=self.shard_size,
                            unit="tokens",
                            desc=f"Shard {shard_index}",
                        )
                    progress_bar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(
                        self.local_dir, f"edufineweb_{split}_{shard_index:06d}"
                    )
                    remainder = self.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count : token_count + remainder] = tokens[
                        :remainder
                    ]
                    self.write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                    token_count = len(tokens) - remainder
            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    self.local_dir, f"edufineweb_{split}_{shard_index:06d}"
                )
                self.write_datafile(filename, all_tokens_np[:token_count])
