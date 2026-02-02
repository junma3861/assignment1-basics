import regex as re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os
from typing import BinaryIO


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
        
        # Remove all special tokens from the text before pre-tokenization
        for special_token in special_tokens:
            chunk = chunk.replace(special_token, "")
        
        tokens = re.findall(PAT, chunk, flags=re.UNICODE)
        for token in tokens:
            byte_token = token.encode("utf-8")
            # Split into individual bytes for BPE processing
            token_freqs[tuple(bytes([b]) for b in byte_token)] += 1
    
    return token_freqs


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = None) -> None:
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
    ## Need to add special tokens "<|endoftext|>"
    vocab["<|endoftext|>"] = 256

    # Set number of processes
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"\nPre-tokenization using {num_processes} processes...")

    # Pre-tokenization pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Parallel pre-tokenization using chunk boundaries from pretokenization_example.py
    with open(input_path, "rb") as f:
        # Get chunk boundaries at special token positions
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n\n"
        boundaries = find_chunk_boundaries(f, num_processes, split_token)
        print(f"Split file into {len(boundaries) - 1} chunks at special token boundaries")
        
        # Prepare arguments for each chunk (start/end pairs)
        chunk_args = [
            (input_path, start, end, special_tokens, PAT)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
    
    # Process chunks in parallel
    token_freqs = defaultdict(int)
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(pretokenize_chunk, chunk_args)
    
    # Merge results from all chunks
    for chunk_freq in chunk_results:
        for token_tuple, freq in chunk_freq.items():
            token_freqs[token_tuple] += freq
    
    print(f"Pre-tokenization complete. Found {len(token_freqs)} unique token sequences.")

    # BPE Training
    merges = []
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
        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)
        new_token = b"".join(most_frequent_pair)
        vocab[new_token] = len(vocab)
        merges.append(most_frequent_pair)
        
        # Print progress
        if len(merges) % 100 == 0 or len(merges) <= 10:
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
    



if __name__ == "__main__":
    inv_vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])
    
    print("=" * 80)
    print("BPE Training Complete!")
    print("=" * 80)
    print(f"\nVocabulary size: {len(inv_vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    print("\n" + "-" * 80)
    print("First 10 merges:")
    print("-" * 80)
    for i, (token1, token2) in enumerate(merges[:10], 1):
        try:
            token1_str = token1.decode('utf-8', errors='replace')
            token2_str = token2.decode('utf-8', errors='replace')
            merged = (token1 + token2).decode('utf-8', errors='replace')
            print(f"{i:2d}. {repr(token1_str):20s} + {repr(token2_str):20s} -> {repr(merged)}")
        except:
            print(f"{i:2d}. {token1!r:20s} + {token2!r:20s}")
    
    print("\n" + "-" * 80)
    print("Sample vocabulary entries (first 20 non-byte tokens):")
    print("-" * 80)
    count = 0
    for idx in sorted(inv_vocab.keys()):
        if idx >= 256 and count < 20:  # Skip single byte tokens
            token = inv_vocab[idx]
            try:
                if isinstance(token, bytes):
                    token_str = token.decode('utf-8', errors='replace')
                else:
                    token_str = token
                print(f"ID {idx:4d}: {repr(token_str)}")
                count += 1
            except:
                print(f"ID {idx:4d}: {token!r}")
                count += 1
    
    print("\n" + "=" * 80)