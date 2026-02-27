#!/usr/bin/env python3
"""
Text generation script using a trained Transformer Language Model.

This script:
1. Loads the tokenizer from BPE vocabulary and merges
2. Loads the latest checkpoint
3. Generates text starting from a prompt
4. Outputs at least 256 tokens or until end-of-text token
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Optional

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import generate
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.adam_w import AdamW


def load_tokenizer(vocab_path: str, merges_path: str) -> Tokenizer:
    """
    Load tokenizer from vocabulary and merges files.
    
    Args:
        vocab_path: Path to vocabulary JSON file
        merges_path: Path to merges text file
    
    Returns:
        Tokenizer instance
    """
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file in a directory.
    
    Prefers checkpoint_final.pt if it exists, otherwise returns the highest numbered checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Look for checkpoint_final.pt first
    final_checkpoint = checkpoint_path / "checkpoint_final.pt"
    if final_checkpoint.exists():
        return str(final_checkpoint)
    
    # Otherwise find the highest numbered checkpoint
    checkpoint_files = sorted(checkpoint_path.glob("checkpoint_iter_*.pt"))
    if checkpoint_files:
        return str(checkpoint_files[-1])
    
    return None


def get_vocab_size_from_vocab(vocab_path: str) -> int:
    """
    Get vocabulary size from vocab.json file.
    
    Args:
        vocab_path: Path to vocabulary JSON file
    
    Returns:
        Vocabulary size
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    return len(vocab_dict)


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained Transformer LM")
    
    # Model and checkpoint paths
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="bpe/TinyStoriesV2/vocab.json",
        help="Path to vocabulary JSON file"
    )
    parser.add_argument(
        "--merges_path",
        type=str,
        default="bpe/TinyStoriesV2/merges.txt",
        help="Path to BPE merges file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/tinystories_bs128_lr3e-3",
        help="Directory containing checkpoints"
    )
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Vocabulary size (auto-detected if not provided)")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344,
                        help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Text Generation with Transformer Language Model")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Determine vocab size
    vocab_size = args.vocab_size
    if vocab_size is None:
        vocab_size = get_vocab_size_from_vocab(args.vocab_path)
        print(f"Auto-detected vocab size: {vocab_size}")
    
    print(f"\nLoading tokenizer from:")
    print(f"  Vocabulary: {args.vocab_path}")
    print(f"  Merges: {args.merges_path}")
    tokenizer = load_tokenizer(args.vocab_path, args.merges_path)
    print(f"Tokenizer loaded. Vocabulary size: {vocab_size}")
    
    # Create model
    print(f"\nCreating model with hyperparameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  context_length: {args.context_length}")
    print(f"  d_model: {args.d_model}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  num_layers: {args.num_layers}")
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers
    )
    model = model.to(args.device)
    print(f"Model created on device: {args.device}")
    
    # Load checkpoint
    checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None:
        print(f"\nERROR: No checkpoint found in {args.checkpoint_dir}")
        return
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Create a dummy optimizer for checkpoint loading
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Load checkpoint with map_location to handle device mismatch
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        iteration = checkpoint_data["iteration"]
        print(f"Checkpoint loaded successfully (iteration {iteration})")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return
    
    # Encode prompt
    print(f"\nPrompt: \"{args.prompt}\"")
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Encoded to {len(prompt_ids)} tokens: {prompt_ids[:20]}{'...' if len(prompt_ids) > 20 else ''}")
    
    # Convert to tensor
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    
    # Get actual end-of-text token ID from tokenizer
    eos_token_string = "<|endoftext|>"
    eos_token_ids = tokenizer.encode(eos_token_string)
    eos_token_id = eos_token_ids[0] if eos_token_ids else vocab_size - 1
    print(f"\nEnd-of-text token ID: {eos_token_id} (decoded: '{tokenizer.decode(eos_token_ids)}')")
    
    # Generate
    print(f"\nGenerating text...")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print("-" * 80)
    
    with torch.no_grad():
        generated_ids = generate(
            model=model,
            prompt_ids=prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,  # Use tokenizer's actual end-of-text token
            device=args.device
        )
    
    # Decode and print
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    print(generated_text)
    print("-" * 80)
    
    # Statistics
    total_tokens = len(generated_ids[0]) - len(prompt_ids)
    print(f"\nGeneration Statistics:")
    print(f"  Original prompt tokens: {len(prompt_ids)}")
    print(f"  Generated tokens: {total_tokens}")
    print(f"  Total tokens: {len(generated_ids[0])}")


if __name__ == "__main__":
    main()
