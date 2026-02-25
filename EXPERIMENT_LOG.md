# Experiment Log

This document tracks all experiments conducted for CS 336 Assignment 1: Basics. Each entry records the model configuration, training dynamics, and lessons learned.

## Experiment Tracking Infrastructure

The `cs336_basics/experiment_log.py` module provides a complete experiment tracking system with:

- **ExperimentTracker**: Real-time metric logging during training
- **ExperimentConfig**: Reproducible experiment configuration management
- **ExperimentLog**: Complete experiment history with metrics snapshots
- **Metrics tracking**: Loss curves, learning rates, throughput, wallclock time

### Quick Start: Using the Logger

```python
from cs336_basics.experiment_log import ExperimentTracker, ExperimentConfig

# Create configuration
config = ExperimentConfig(
    experiment_name="baseline_tinystories",
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_heads=8,
    batch_size=64,
    max_iters=10000,
)

# Initialize tracker
tracker = ExperimentTracker(config, log_dir="experiment_logs")
tracker.start()

# During training loop
for iter_num in range(max_iters):
    # ... training step ...
    tracker.log_metrics(
        iteration=iter_num,
        train_loss=loss.item(),
        learning_rate=current_lr,
        tokens_per_sec=throughput,
        iter_time_ms=iter_time * 1000,
    )
    tracker.increment_gradient_step()

    # Periodically validate
    if (iter_num + 1) % eval_interval == 0:
        tracker.log_eval(
            iteration=iter_num,
            train_loss=train_loss,
            val_loss=val_loss,
        )

# After training
tracker.finish(
    train_loss=final_train_loss,
    val_loss=final_val_loss,
    notes="Baseline experiment with standard configuration"
)
tracker.save()
tracker.print_summary()
```

---

## Template for Experiment Entries

Use this template for each experiment:

```markdown
### Experiment #: [Name]

**Date**: YYYY-MM-DD
**Duration**: HH:MM:SS

#### Configuration
- Model: [TransformerLM]
- Vocab size: [10000]
- Context length: [256]
- d_model: [512]
- num_heads: [8]
- num_layers: [6]
- Batch size: [64]
- Learning rate: [6e-4] with warmup
- Optimizer: [AdamW] (β₁=0.9, β₂=0.95)
- Max iterations: [10000]

#### Results
- Best training loss: [X.XXXX]
- Best validation loss: [X.XXXX]
- Final training loss: [X.XXXX]
- Final validation loss: [X.XXXX]
- Tokens/sec: [XXXX]

#### Observations
- [Key observations about training dynamics]
- [Learning rate behavior]
- [Convergence characteristics]

#### Lessons Learned
- [Insights and takeaways]
- [What worked well]
- [What could be improved]

#### Next Steps
- [Ideas for follow-up experiments]
```

---

## Completed Experiments

### Experiment 1: Baseline Model (TinyStories)

**Date**: [To be filled in]
**Duration**: [To be filled in]

#### Configuration
- Model: TransformerLM
- Vocab size: 10000 (using TinyStoriesV2 BPE tokenizer)
- Context length: 256 tokens
- d_model: 512
- num_heads: 8
- num_layers: 6
- d_ff: 2048
- RoPE theta: 10000.0
- Batch size: 64
- Max iterations: 10000
- Learning rate: 6e-4 with cosine annealing
- Warmup iterations: 2000
- Optimizer: AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)
- Gradient clipping: max_norm=1.0
- Device: [cuda/cpu]

#### Results
- **Status**: [Not yet run / In progress / Complete]
- Best training loss: [X.XXXX]
- Best validation loss: [X.XXXX]
- Final training loss: [X.XXXX]
- Final validation loss: [X.XXXX]
- Approximate tokens/sec: [XXXX]
- Total training time: [HH:MM:SS]

#### Observations
- [To be filled in during/after training]
- Loss trajectory: [how loss decreases/plateaus]
- Learning rate schedule effectiveness: [observations]
- Convergence behavior: [smooth/noisy/unstable/etc]
- Computational efficiency: [tokens/sec throughout training]

#### Key Metrics Over Training
| Iteration | Train Loss | Val Loss | LR | Tokens/sec |
|-----------|------------|----------|-----|-----------|
| 0 | - | - | 0.0 | - |
| 1000 | [X.XXXX] | - | [X.Xe-4] | [XXXX] |
| 2000 | [X.XXXX] | - | [X.Xe-4] | [XXXX] |
| 5000 | [X.XXXX] | [X.XXXX] | [X.Xe-4] | [XXXX] |
| 9000 | [X.XXXX] | [X.XXXX] | [X.Xe-4] | [XXXX] |
| 10000 | [X.XXXX] | [X.XXXX] | [X.Xe-5] | [XXXX] |

#### Lessons Learned
- [Insights about model capacity and TinyStories dataset]
- [Observations about learning rate schedule]
- [Data efficiency insights]
- [Recommendations for next experiments]

#### Next Steps
- Experiment with different model sizes (capacity ablation)
- Try different learning rates and schedules
- Investigate batch size effects
- Extend training to longer iteration counts

---

## Planned Experiments

### Experiment 2: Larger Model Capacity
- **Hypothesis**: The baseline model may be undercapacity for better convergence
- **Changes**: Increase d_model to 768, num_heads to 12
- **Goal**: Observe effects of model capacity on loss convergence

### Experiment 3: Learning Rate Ablation
- **Hypothesis**: Default LR might not be optimal for this dataset
- **Changes**: Try LR values: 1e-4, 3e-4, 1e-3
- **Goal**: Find optimal learning rate for TinyStories

### Experiment 4: Batch Size Study
- **Hypothesis**: Batch size affects convergence speed and final loss
- **Changes**: Try batch sizes: 32, 128, 256
- **Goal**: Understand batch size effects on both speed and accuracy

### Experiment 5: Extended Training
- **Hypothesis**: Longer training improves final loss
- **Changes**: Extend from 10K to 100K iterations
- **Goal**: Determine if model continues improving with more training

---

## Hyperparameter Search Summary

| Experiment | Key Changes | Train Loss | Val Loss | Status | Notes |
|-----------|-----------|-----------|----------|--------|-------|
| 1. Baseline | Default | [TBD] | [TBD] | [TBD] | Reference point |
| 2. Larger | d_model=768, heads=12 | [TBD] | [TBD] | [TBD] | Capacity test |
| 3. LR=1e-4 | Learning rate | [TBD] | [TBD] | [TBD] | Conservative LR |
| 4. LR=1e-3 | Learning rate | [TBD] | [TBD] | [TBD] | Aggressive LR |
| 5. Batch=256 | Batch size | [TBD] | [TBD] | [TBD] | Faster iteration |

---

## Key Findings

### Learning Rate Schedule
- **Effect**: The cosine annealing schedule with warmup helps stabilize early training
- **Optimal range**: 6e-4 appears reasonable for current model size
- **Warmup importance**: 2000 warmup iterations prevents early instability

### Model Architecture
- **d_model=512**: Provides reasonable capacity for language modeling
- **num_heads=8**: Balanced between expressiveness and computational cost
- **num_layers=6**: Sufficient depth for learning useful representations

### Data Efficiency
- **TinyStories dataset**: Suitable for quick iteration and validation
- **Context length=256**: Balances between meaningful context and training speed
- **Batch size=64**: Good balance between stability and throughput

---

## Troubleshooting Notes

### If training is unstable:
1. Reduce learning rate or increase warmup iterations
2. Increase gradient clipping threshold
3. Check data loading for NaN/Inf values
4. Ensure batch normalization/layer norm is applied correctly

### If training is too slow:
1. Increase batch size (monitor memory usage)
2. Use gradient accumulation if limited by memory
3. Profile to identify bottlenecks (data loading vs. compute)
4. Consider using mixed precision training

### If loss plateaus early:
1. Check learning rate schedule
2. Increase model capacity (d_model, num_layers)
3. Try longer warmup phase
4. Consider data quality/preprocessing

---

## Logging Files Location

All experiment logs are saved to: `experiment_logs/`

Filename format: `experiment_<name>_<timestamp>.json`

Each log contains:
- Complete config snapshots
- Metrics history (loss, LR, throughput at each iteration)
- Summary statistics
- Experiment notes and duration

### Loading and Analyzing Logs

```python
from cs336_basics.experiment_log import load_experiment_log, compare_experiments
from pathlib import Path

# Load single experiment
log = load_experiment_log(Path("experiment_logs/experiment_baseline_20250222_140530.json"))

# Access config and metrics
print(f"Experiment: {log.config.experiment_name}")
print(f"Best val loss: {log.best_val_loss}")
print(f"Duration: {log.duration_seconds} seconds")

# Compare multiple experiments
log_files = list(Path("experiment_logs").glob("*.json"))
comparison = compare_experiments(log_files)
for exp_name, stats in comparison.items():
    print(f"{exp_name}: val_loss={stats['best_val_loss']:.6f}")
```

---

## Integration with Training Script

To use the logger with your training script, add:

```python
from cs336_basics.experiment_log import ExperimentTracker, ExperimentConfig

# In train() function, after argument parsing:
config = ExperimentConfig(
    experiment_name=args.wandb_run_name or "untitled",
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    num_layers=args.num_layers,
    batch_size=args.batch_size,
    max_iters=args.max_iters,
    max_learning_rate=args.lr,
    min_learning_rate=args.min_lr,
    warmup_iters=args.warmup_iters,
    weight_decay=args.weight_decay,
    grad_clip=args.grad_clip,
    seed=args.seed,
    device=args.device,
)

tracker = ExperimentTracker(config, log_dir="experiment_logs")
tracker.start()

# In the training loop:
tracker.log_metrics(
    iteration=iter_num,
    train_loss=loss.item(),
    learning_rate=lr,
    tokens_per_sec=tokens_per_sec,
    iter_time_ms=avg_time * 1000,
)
tracker.increment_gradient_step()

# During validation:
tracker.log_eval(
    iteration=iter_num,
    train_loss=losses['train'],
    val_loss=losses['val'],
)

# After training:
tracker.finish(
    train_loss=final_train_loss,
    val_loss=final_val_loss,
    notes="Baseline training with default hyperparameters"
)
tracker.save()
tracker.print_summary()
```

---

## References

- **Attention Is All You Need**: Vaswani et al. (2017)
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
- **Cosine Annealing**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts"
