import argparse
import os

from gpt.utils.fineweb import FineWebEdu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="artifacts/data/fine_web_edu")
    parser.add_argument("--remote_name", type=str, default="sample-10BT")
    parser.add_argument("--shard_size", type=int, default=int(1e8))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    fwe = FineWebEdu(args.local_dir, args.remote_name, args.shard_size)
    fwe.write_shards()


if __name__ == "__main__":
    main()
