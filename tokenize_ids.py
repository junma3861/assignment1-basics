"""
Tokenize TinyStories data and write one token ID per line.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def encode_to_file(tokenizer: Tokenizer, src: Path, dst: Path) -> None:
    print(f"Processing {src} (streaming mode)...")
    
    start_time = time.perf_counter()
    token_count = 0
    char_count = 0
    
    with open(src, "r", encoding="utf-8") as input_file, open(dst, "w", encoding="utf-8") as output_file:
        # Use encode_iterable for memory-efficient processing
        for i, token_id in enumerate(tokenizer.encode_iterable(input_file)):
            output_file.write(f"{token_id}\n")
            token_count += 1
            
            # Show progress every 100K tokens
            if token_count % 100_000 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Processed {token_count:,} tokens in {elapsed:.1f}s ({token_count/elapsed:.0f} tok/s)")
    
    # Count characters for final stats
    with open(src, "r", encoding="utf-8") as f:
        char_count = sum(len(line) for line in f)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print(f"Done! {src} -> {dst}")
    print(f"  Characters: {char_count:,}")
    print(f"  Tokens: {token_count:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {token_count / total_time:.0f} tokens/s")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize TinyStories data to .ids files."
    )
    parser.add_argument(
        "--vocab",
        default="bpe/TinyStoriesV2/vocab.json",
        help="Path to vocab.json",
    )
    parser.add_argument(
        "--merges",
        default="bpe/TinyStoriesV2/merges.txt",
        help="Path to merges.txt",
    )
    parser.add_argument(
        "--train",
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to train text file",
    )
    parser.add_argument(
        "--valid",
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="Path to valid text file",
    )
    parser.add_argument(
        "--train-out",
        default="data/TinyStoriesV2-GPT4-train.ids",
        help="Output .ids file for train",
    )
    parser.add_argument(
        "--valid-out",
        default="data/TinyStoriesV2-GPT4-valid.ids",
        help="Output .ids file for valid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges)

    encode_to_file(tokenizer, Path(args.train), Path(args.train_out))
    encode_to_file(tokenizer, Path(args.valid), Path(args.valid_out))


if __name__ == "__main__":
    main()
