import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(args):
    """
    Pre-tokenize a chunk of the file and return token frequencies.
    Worker function for parallel processing.
    """
    input_path, start, end, special_tokens, PAT = args
    
    token_freqs = defaultdict(int)
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Normalize Windows newlines to match snapshot expectations
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

        if special_tokens:
            special_pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
            parts = re.split(special_pattern, chunk)
        else:
            parts = [chunk]

        for part in parts:
            if part == "":
                continue
            if part in special_tokens:
                token_freqs[(part.encode("utf-8"),)] += 1
                continue
            tokens = re.findall(PAT, part, flags=re.UNICODE)
            for token in tokens:
                byte_token = token.encode("utf-8")
                # Split into individual bytes for BPE processing
                token_freqs[tuple(bytes([b]) for b in byte_token)] += 1
    
    return token_freqs


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
    verbose: bool | None = None,
) -> None:
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given input file.

    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired vocabulary size for the tokenizer.
        special_tokens (list[str]): List of special tokens to include in the tokenizer.
        num_processes (int, optional): Number of processes for parallel pre-tokenization.
                                      If None, uses cpu_count().

    Returns:
        None
    """
    
    # Initialize vocabulary with single byte tokens
    vocab = {bytes([i]): i for i in range(256)}
    
    # Add special tokens to vocabulary
    for special_token in special_tokens:
        vocab[special_token.encode("utf-8")] = len(vocab)

    # Set number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    # Pre-tokenization pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Parallel pre-tokenization using chunk boundaries from pretokenization_example.py
    with open(input_path, "rb") as f:
        # Get chunk boundaries at special token positions
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n\n"
        # For small files, multiprocessing overhead dominates; fall back to single-process.
        file_size = f.seek(0, os.SEEK_END)
        f.seek(0)
        if verbose is None:
            verbose = file_size >= 1_000_000
        if file_size < 1_000_000:
            num_processes = 1
        if verbose:
            print(f"\nPre-tokenization using {num_processes} processes...")
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
        if verbose:
            print(f"Split file into {len(boundaries) - 1} chunks at special token boundaries")
        
        # Prepare arguments for each chunk (start/end pairs)
        chunk_args = [
            (input_path, start, end, special_tokens, PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
    
    # Process chunks (parallel only when beneficial)
    token_freqs = defaultdict(int)
    if num_processes == 1 or len(chunk_args) <= 1:
        chunk_results = [pretokenize_chunk(args) for args in chunk_args]
    else:
        with Pool(processes=num_processes) as pool:
            chunk_results = pool.map(pretokenize_chunk, chunk_args)
    
    # Merge results from all chunks
    for chunk_freq in chunk_results:
        for token_tuple, freq in chunk_freq.items():
            token_freqs[token_tuple] += freq
    
    if verbose:
        print(f"Pre-tokenization complete. Found {len(token_freqs)} unique token sequences.")

    # BPE Training
    merges = []
    if verbose:
        print(f"\nStarting BPE training: vocab_size={len(vocab)}, target={vocab_size}")
        print("=" * 80)
    
    while len(vocab) < vocab_size:
        pair_freqs = defaultdict(int)

        # Count frequencies of adjacent token pairs
        for token_tuple, freq in token_freqs.items():
            for i in range(len(token_tuple) - 1):
                pair = (token_tuple[i], token_tuple[i + 1])
                pair_freqs[pair] += freq

        if not pair_freqs:
            break  # No more pairs to merge

        # Find the most frequent pair
        # When there are ties in frequency, use lexicographic ordering as tiebreaker
        most_frequent_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        new_token = b"".join(most_frequent_pair)
        vocab[new_token] = len(vocab)
        merges.append(most_frequent_pair)
        
        # Print progress
        if verbose and (len(merges) % 100 == 0 or len(merges) <= 10):
            try:
                token1_str = most_frequent_pair[0].decode('utf-8', errors='replace')
                token2_str = most_frequent_pair[1].decode('utf-8', errors='replace')
                merged_str = new_token.decode('utf-8', errors='replace')
                print(f"Merge {len(merges):5d}/{vocab_size-256}: {repr(token1_str)} + {repr(token2_str)} -> {repr(merged_str)} (freq={pair_freqs[most_frequent_pair]})")
            except:
                print(f"Merge {len(merges):5d}/{vocab_size-256}: freq={pair_freqs[most_frequent_pair]}")

        # Update token frequencies with the new merged token
        new_token_tuple = (new_token,)
        updated_token_freqs = defaultdict(int)

        for token_tuple, freq in token_freqs.items():
            new_tuple = []
            i = 0
            while i < len(token_tuple):
                if (i < len(token_tuple) - 1 and
                        token_tuple[i] == most_frequent_pair[0] and
                        token_tuple[i + 1] == most_frequent_pair[1]):
                    new_tuple.append(new_token)
                    i += 2
                else:
                    new_tuple.append(token_tuple[i])
                    i += 1
            updated_token_freqs[tuple(new_tuple)] += freq

        token_freqs = updated_token_freqs

    # inverse vocab for easy lookup
    inv_vocab = {v: k for k, v in vocab.items()}

    return inv_vocab, merges


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def _bytes_to_gpt2_str(token_bytes: bytes, b2u: dict[int, str]) -> str:
    return "".join(b2u[b] for b in token_bytes)


def save_bpe_artifacts(
    inv_vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | os.PathLike,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    b2u = bytes_to_unicode()

    vocab_json = {
        _bytes_to_gpt2_str(token_bytes, b2u): token_id
        for token_id, token_bytes in inv_vocab.items()
    }

    vocab_path = output_path / "vocab.json"
    merges_path = output_path / "merges.txt"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for token1, token2 in merges:
            f.write(f"{_bytes_to_gpt2_str(token1, b2u)} {_bytes_to_gpt2_str(token2, b2u)}\n")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and optionally save artifacts.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to training text file.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Total vocabulary size.")
    parser.add_argument(
        "--special_token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to add (repeatable).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bpe_output",
        help="Output directory for vocab.json and merges.txt.",
    )
    args = parser.parse_args()

    inv_vocab, merges = train_bpe(
        args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
    )

    save_bpe_artifacts(inv_vocab, merges, args.output_dir)