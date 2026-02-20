"""
Training script for Transformer Language Model.

This script provides a configurable training loop with the following features:
- Configurable model and optimizer hyperparameters via command-line arguments
- Memory-efficient data loading using np.memmap
- Checkpoint saving and loading
- Periodic logging of training and validation metrics
- Optional Weights & Biases integration
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adam_w import AdamW
from cs336_basics.data_loading import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import clip_gradients


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # Data arguments
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training data (.ids file)")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Path to validation data (.ids file)")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Optimizer hyperparameters
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay coefficient")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Adam epsilon")
    
    # Learning rate schedule
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="Number of warmup iterations")
    parser.add_argument("--max_iters", type=int, default=100000,
                        help="Maximum number of training iterations")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    
    # Checkpointing and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=5000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate on validation set every N iterations")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log training metrics every N iterations")
    parser.add_argument("--eval_iters", type=int, default=100,
                        help="Number of batches to use for validation")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")
    
    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_memmap(data_path: str) -> np.ndarray:
    """
    Load data using memory-mapped file for efficient large dataset handling.
    
    Args:
        data_path: Path to .ids file containing token IDs
        
    Returns:
        Memory-mapped numpy array
    """
    # Load as memory-mapped array for efficiency
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    return data


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str
) -> dict[str, float]:
    """
    Estimate loss on training and validation sets.
    
    Args:
        model: The model to evaluate
        train_data: Training dataset
        val_data: Validation dataset
        batch_size: Batch size for evaluation
        context_length: Context length
        eval_iters: Number of batches to evaluate
        device: Device to run evaluation on
        
    Returns:
        Dictionary with 'train' and 'val' loss values
    """
    model.eval()
    losses = {}
    
    for split_name, data in [('train', train_data), ('val', val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, context_length, device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            split_losses.append(loss.item())
        losses[split_name] = np.mean(split_losses)
    
    model.train()
    return losses


def train(args):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize Weights & Biases if requested
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            print("Weights & Biases logging enabled")
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb'")
            args.use_wandb = False
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Load data using memory mapping for efficiency
    print("Loading data with memory mapping...")
    train_data = load_data_memmap(args.train_data_path)
    val_data = load_data_memmap(args.val_data_path)
    print(f"Train data size: {len(train_data):,} tokens")
    print(f"Validation data size: {len(val_data):,} tokens")
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta
    )
    model = model.to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from is not None:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # Training loop
    print(f"\nStarting training from iteration {start_iter}...")
    print(f"Device: {args.device}")
    print(f"Max iterations: {args.max_iters}")
    print("-" * 80)
    
    model.train()
    train_losses = []
    iter_times = []
    
    for iter_num in range(start_iter, args.max_iters):
        iter_start_time = time.time()
        
        # Update learning rate using cosine schedule with warmup
        lr = get_lr_cosine_schedule(
            iter_num,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Sample a batch of data
        inputs, targets = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            args.device
        )
        
        # Forward pass
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            clip_gradients(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Track metrics
        train_losses.append(loss.item())
        iter_time = time.time() - iter_start_time
        iter_times.append(iter_time)
        
        # Logging
        if (iter_num + 1) % args.log_interval == 0:
            avg_loss = np.mean(train_losses[-args.log_interval:])
            avg_time = np.mean(iter_times[-args.log_interval:])
            tokens_per_sec = (args.batch_size * args.context_length) / avg_time
            
            print(f"Iter {iter_num + 1:6d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {avg_time*1000:.1f}ms | "
                  f"Tokens/sec: {tokens_per_sec:.0f}")
            
            if args.use_wandb and wandb_run is not None:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/iter_time_ms": avg_time * 1000,
                    "iteration": iter_num + 1
                })
        
        # Validation
        if (iter_num + 1) % args.eval_interval == 0:
            print("-" * 80)
            print(f"Evaluating at iteration {iter_num + 1}...")
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.context_length,
                args.eval_iters,
                args.device
            )
            print(f"Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")
            print("-" * 80)
            
            if args.use_wandb and wandb_run is not None:
                wandb.log({
                    "eval/train_loss": losses['train'],
                    "eval/val_loss": losses['val'],
                    "iteration": iter_num + 1
                })
        
        # Save checkpoint
        if (iter_num + 1) % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num + 1}.pt"
            save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"\nTraining complete! Final checkpoint saved to {final_checkpoint_path}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    losses = estimate_loss(
        model,
        train_data,
        val_data,
        args.batch_size,
        args.context_length,
        args.eval_iters,
        args.device
    )
    print(f"Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")
    
    if args.use_wandb and wandb_run is not None:
        wandb.log({
            "final/train_loss": losses['train'],
            "final/val_loss": losses['val'],
        })
        wandb_run.finish()


def main():
    """Entry point for the training script."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
