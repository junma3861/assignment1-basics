"""
BPE Tokenizer implementation for encoding and decoding text.
"""

from __future__ import annotations

import json
import regex as re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional


class Tokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer that encodes text into token IDs
    and decodes token IDs back into text.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens.

        Args:
            vocab: dict[int, bytes] - mapping from token ID to token bytes
            merges: list[tuple[bytes, bytes]] - ordered list of BPE merges
            special_tokens: list[str] | None - special tokens to preserve during tokenization
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens else set()
        
        # Create inverse vocabulary for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Create a mapping from merge pair to its rank (priority)
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        
        # Add special tokens to vocabulary if they're not already there
        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.inv_vocab:
                    token_id = len(self.vocab)
                    self.vocab[token_id] = special_token_bytes
                    self.inv_vocab[special_token_bytes] = token_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Construct a tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath: Path to vocabulary JSON file (token string -> token ID)
            merges_filepath: Path to merges text file
            special_tokens: list[str] | None - special tokens to preserve

        Returns:
            Tokenizer: A tokenizer instance
        """
        # Load vocabulary from JSON
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        
        # Convert from GPT-2 string representation to bytes
        # We need to reverse the bytes_to_unicode mapping
        b2u = _bytes_to_unicode()
        u2b = {v: k for k, v in b2u.items()}
        
        vocab = {}
        for token_str, token_id in vocab_dict.items():
            # Convert the string representation back to bytes
            token_bytes = bytes(u2b[c] for c in token_str)
            vocab[token_id] = token_bytes
        
        # Load merges from text file
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line and not line.startswith("#"):
                    parts = line.split(" ")
                    if len(parts) == 2:
                        merge_token_1_str = parts[0]
                        merge_token_2_str = parts[1]
                        # Convert from string representation back to bytes
                        merge_token_1_bytes = bytes(u2b[c] for c in merge_token_1_str)
                        merge_token_2_bytes = bytes(u2b[c] for c in merge_token_2_str)
                        merges.append((merge_token_1_bytes, merge_token_2_bytes))
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: str - text to encode

        Returns:
            list[int]: list of token IDs
        """
        # Handle special tokens first
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]
        
        token_ids = []
        
        for part in parts:
            if part in self.special_tokens:
                # Special token - add it directly
                special_token_bytes = part.encode("utf-8")
                token_id = self.inv_vocab[special_token_bytes]
                token_ids.append(token_id)
            elif part:
                # Regular text - tokenize using BPE
                token_ids.extend(self._encode_regular(part))
        
        return token_ids

    def _encode_regular(self, text: str) -> list[int]:
        """
        Encode regular text (non-special-token) using BPE.

        Args:
            text: str - text to encode

        Returns:
            list[int]: list of token IDs
        """
        # Tokenize using the GPT-2 regex pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens = re.findall(PAT, text, flags=re.UNICODE)
        
        # Convert each token to bytes and then to BPE subword tokens
        bpe_tokens = []
        for token in tokens:
            token_bytes = token.encode("utf-8")
            # Start with individual bytes
            byte_tokens = [[b] for b in token_bytes]
            
            # Apply BPE merges
            while len(byte_tokens) > 1:
                # Find the best merge (lowest rank)
                best_merge_idx = -1
                best_merge_rank = float("inf")
                
                for i in range(len(byte_tokens) - 1):
                    # Try to create the pair
                    token1_bytes = bytes(byte_tokens[i])
                    token2_bytes = bytes(byte_tokens[i + 1])
                    pair = (token1_bytes, token2_bytes)
                    
                    # Check if this pair is in our merge ranks
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_merge_rank:
                            best_merge_rank = rank
                            best_merge_idx = i
                
                # If no merge found, break
                if best_merge_idx == -1:
                    break
                
                # Perform the merge
                token1_bytes = bytes(byte_tokens[best_merge_idx])
                token2_bytes = bytes(byte_tokens[best_merge_idx + 1])
                merged_bytes = token1_bytes + token2_bytes
                
                # Replace the two tokens with the merged token
                byte_tokens = (
                    byte_tokens[:best_merge_idx]
                    + [[b for b in merged_bytes]]
                    + byte_tokens[best_merge_idx + 2 :]
                )
            
            # Convert the final byte tokens to token IDs
            for byte_token in byte_tokens:
                token_bytes = bytes(byte_token)
                if token_bytes in self.inv_vocab:
                    bpe_tokens.append(self.inv_vocab[token_bytes])
        
        return bpe_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (e.g., file handle) into token IDs.
        This is memory-efficient for large files that cannot fit in memory.

        Args:
            iterable: Iterable[str] - iterable of strings to encode

        Yields:
            int: token IDs one at a time
        """
        # Process the iterable line by line to avoid loading everything into memory
        for line in iterable:
            # Get the token IDs for this line
            token_ids = self.encode(line)
            # Yield them one at a time
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: list[int] - list of token IDs to decode

        Returns:
            str: decoded text
        """
        # Get the bytes for each token ID
        token_bytes_list = []
        for token_id in ids:
            if token_id in self.vocab:
                token_bytes_list.append(self.vocab[token_id])
        
        # Concatenate all the bytes
        full_bytes = b"".join(token_bytes_list)
        
        # Decode to string
        try:
            return full_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Handle cases where we can't decode as UTF-8
            return full_bytes.decode("utf-8", errors="replace")


def _bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (0-255) to a printable unicode string.
    This matches the GPT-2 encoding scheme.
    """
    # These 188 integers can be used as-is (printable ASCII + extended ASCII)
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    
    # Map the remaining 68 bytes to shifted unicode characters
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))
