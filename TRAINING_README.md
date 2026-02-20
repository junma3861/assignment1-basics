# Training Script Documentation

## Overview

The `training_together.py` script provides a comprehensive training loop for the Transformer Language Model with all essential features for production-grade training.

## Key Features

### ✅ Configurable Hyperparameters
- Full control over model architecture (vocab size, layers, attention heads, dimensions)
- Optimizer settings (learning rate, weight decay, betas, epsilon)
- Training parameters (batch size, iterations, gradient clipping)
- All configurable via command-line arguments

### ✅ Memory-Efficient Data Loading
- Uses `np.memmap` for loading large datasets without loading entire files into RAM
- Efficient batch sampling from memory-mapped arrays
- Supports multi-GB datasets with minimal memory footprint

### ✅ Checkpoint Management
- Save checkpoints at configurable intervals
- Resume training from any checkpoint
- Checkpoints include model state, optimizer state, and iteration number
- Automatic checkpoint directory creation

### ✅ Comprehensive Logging
- Console logging of training metrics (loss, learning rate, throughput)
- Periodic validation evaluation
- Optional Weights & Biases integration for experiment tracking
- Configurable logging intervals

## Usage

### Basic Training

```bash
python -m cs336_basics.training_together \
    --train_data_path data/TinyStoriesV2-GPT4-train.ids \
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids \
    --vocab_size 10000 \
    --max_iters 10000
```

### Full Configuration Example

```bash
python -m cs336_basics.training_together \
    --train_data_path data/TinyStoriesV2-GPT4-train.ids \
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids \
    --vocab_size 10000 \
    --context_length 512 \
    --d_model 768 \
    --num_heads 12 \
    --d_ff 3072 \
    --num_layers 12 \
    --lr 3e-4 \
    --min_lr 3e-5 \
    --weight_decay 0.1 \
    --batch_size 32 \
    --max_iters 100000 \
    --checkpoint_dir checkpoints/my_run \
    --use_wandb \
    --wandb_project my-project
```

### Resume from Checkpoint

```bash
python -m cs336_basics.training_together \
    --train_data_path data/TinyStoriesV2-GPT4-train.ids \
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids \
    --resume_from checkpoints/my_run/checkpoint_iter_5000.pt \
    --max_iters 20000
```

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--train_data_path` | Path to training data (.ids file) |
| `--val_data_path` | Path to validation data (.ids file) |

### Model Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--vocab_size` | 10000 | Vocabulary size |
| `--context_length` | 256 | Maximum context length |
| `--d_model` | 512 | Model dimension |
| `--num_heads` | 8 | Number of attention heads |
| `--d_ff` | 2048 | Feed-forward dimension |
| `--num_layers` | 6 | Number of transformer layers |
| `--rope_theta` | 10000.0 | RoPE theta parameter |

### Optimizer Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 6e-4 | Maximum learning rate |
| `--min_lr` | 6e-5 | Minimum learning rate |
| `--weight_decay` | 0.1 | Weight decay coefficient |
| `--beta1` | 0.9 | Adam beta1 |
| `--beta2` | 0.95 | Adam beta2 |
| `--eps` | 1e-8 | Adam epsilon |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 64 | Batch size for training |
| `--max_iters` | 100000 | Maximum number of training iterations |
| `--warmup_iters` | 2000 | Number of warmup iterations |
| `--grad_clip` | 1.0 | Gradient clipping max norm |

### Checkpointing & Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | checkpoints | Directory to save checkpoints |
| `--checkpoint_interval` | 5000 | Save checkpoint every N iterations |
| `--eval_interval` | 500 | Evaluate every N iterations |
| `--log_interval` | 100 | Log metrics every N iterations |
| `--eval_iters` | 100 | Number of batches for validation |
| `--resume_from` | None | Path to checkpoint to resume from |

### Other Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | auto | Device to train on (cuda/cpu) |
| `--seed` | 42 | Random seed for reproducibility |
| `--use_wandb` | False | Enable Weights & Biases logging |
| `--wandb_project` | transformer-lm | W&B project name |
| `--wandb_run_name` | None | W&B run name |

## Training Loop Details

The training script implements the following workflow:

1. **Initialization**
   - Set random seeds for reproducibility
   - Load training and validation data using memory mapping
   - Initialize model and move to device
   - Initialize AdamW optimizer
   - Optionally resume from checkpoint

2. **Training Loop** (for each iteration)
   - Update learning rate using cosine schedule with warmup
   - Sample a batch of data
   - Forward pass through model
   - Compute cross-entropy loss
   - Backward pass
   - Gradient clipping
   - Optimizer step
   - Log metrics (every `log_interval` iterations)

3. **Periodic Validation** (every `eval_interval` iterations)
   - Evaluate on both training and validation sets
   - Log evaluation metrics
   - Continue training

4. **Checkpointing** (every `checkpoint_interval` iterations)
   - Save model state, optimizer state, and iteration number
   - Checkpoints can be used to resume training

5. **Final Evaluation**
   - After training completes, perform final evaluation
   - Save final checkpoint

## Output

### Console Output

```
Loading data with memory mapping...
Train data size: 47,483,647 tokens
Validation data size: 2,097,152 tokens

Initializing model...
Total parameters: 52,428,800
Trainable parameters: 52,428,800

Starting training from iteration 0...
Device: cuda
Max iterations: 100000
--------------------------------------------------------------------------------
Iter    100 | Loss: 6.2341 | LR: 3.00e-05 | Time: 145.3ms | Tokens/sec: 113024
Iter    200 | Loss: 5.8123 | LR: 6.00e-05 | Time: 143.1ms | Tokens/sec: 114801
...
--------------------------------------------------------------------------------
Evaluating at iteration 500...
Train loss: 5.1234 | Val loss: 5.2456
--------------------------------------------------------------------------------
Saved checkpoint to checkpoints/run1/checkpoint_iter_5000.pt
```

### Weights & Biases Integration

When `--use_wandb` is enabled, the following metrics are logged:

- **Training metrics** (every `log_interval`)
  - `train/loss`: Training loss
  - `train/learning_rate`: Current learning rate
  - `train/tokens_per_sec`: Training throughput
  - `train/iter_time_ms`: Iteration time in ms

- **Evaluation metrics** (every `eval_interval`)
  - `eval/train_loss`: Training set loss
  - `eval/val_loss`: Validation set loss

- **Final metrics**
  - `final/train_loss`: Final training loss
  - `final/val_loss`: Final validation loss

## Example Workflows

### Quick Test Run

```bash
# Small model, short training for testing
python -m cs336_basics.training_together \
    --train_data_path data/TinyStoriesV2-GPT4-train.ids \
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids \
    --vocab_size 10000 \
    --d_model 256 \
    --num_layers 4 \
    --batch_size 32 \
    --max_iters 1000
```

### Production Training

```bash
# Large model with full features
python -m cs336_basics.training_together \
    --train_data_path data/TinyStoriesV2-GPT4-train.ids \
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids \
    --vocab_size 10000 \
    --context_length 1024 \
    --d_model 1024 \
    --num_heads 16 \
    --d_ff 4096 \
    --num_layers 24 \
    --lr 2e-4 \
    --min_lr 2e-5 \
    --batch_size 16 \
    --max_iters 200000 \
    --checkpoint_interval 2500 \
    --eval_interval 250 \
    --checkpoint_dir checkpoints/large_model \
    --use_wandb \
    --wandb_project cs336-final \
    --wandb_run_name large-model-v1
```

## Tips

1. **Memory Management**: If you run out of GPU memory, try:
   - Reducing `--batch_size`
   - Reducing `--context_length`
   - Reducing model size (`--d_model`, `--d_ff`, `--num_layers`)

2. **Learning Rate**: A good starting learning rate depends on model size:
   - Small models (< 50M params): 6e-4
   - Medium models (50-500M params): 3e-4
   - Large models (> 500M params): 1e-4 to 2e-4

3. **Warmup**: Set `--warmup_iters` to ~2-5% of `--max_iters`

4. **Checkpointing**: Balance checkpoint frequency with disk space:
   - More frequent = safer but uses more disk
   - Less frequent = risky but saves disk space

5. **Validation**: Set `--eval_interval` based on training speed:
   - Fast training (small model): Every 100-500 iterations
   - Slow training (large model): Every 500-2000 iterations
