# Experiment Tracking Infrastructure Guide

## Overview

This guide explains how to use the experiment tracking infrastructure to monitor your CS 336 training experiments. The system tracks loss curves, learning rates, throughput, and wallclock time across training steps.

## Components

### 1. **experiment_log.py** - Core Logging System

The main module providing:

- **ExperimentConfig**: Dataclass for reproducible experiment configuration
- **MetricsSnapshot**: Single point-in-time metric measurement
- **ExperimentLog**: Complete experiment history
- **ExperimentTracker**: Main interface for tracking experiments

### 2. **experiment_analysis.py** - Analysis Tools

Utilities for analyzing results:

- **ExperimentAnalyzer**: Load and compare multiple experiments
- Summary statistics generation
- CSV export for external analysis
- Comparison tables

### 3. **experiment_tracking_example.py** - Usage Examples

Demonstrates:

- Basic tracking setup
- Multiple experiment runs
- Result analysis and comparison

### 4. **EXPERIMENT_LOG.md** - Experiment Documentation

Document for manually recording:

- Experiment results
- Observations and insights
- Planned experiments
- Key findings

## Quick Start

### Step 1: Import and Configure

```python
from cs336_basics.experiment_log import ExperimentTracker, ExperimentConfig

# Create a configuration with your hyperparameters
config = ExperimentConfig(
    experiment_name="my_experiment",
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_heads=8,
    num_layers=6,
    batch_size=64,
    max_iters=100000,
    max_learning_rate=6e-4,
    min_learning_rate=6e-5,
    warmup_iters=2000,
)
```

### Step 2: Initialize Tracker

```python
tracker = ExperimentTracker(config, log_dir="experiment_logs")
tracker.start()
```

### Step 3: Log During Training

```python
for iter_num in range(max_iters):
    # ... your training step ...
    
    # Log metrics every iteration (or at regular intervals)
    tracker.log_metrics(
        iteration=iter_num,
        train_loss=loss.item(),
        learning_rate=current_lr,
        tokens_per_sec=batch_size * context_length / iter_time,
        iter_time_ms=iter_time * 1000,
    )
    
    # Important: increment gradient step counter
    tracker.increment_gradient_step()
    
    # Periodically log validation metrics
    if (iter_num + 1) % eval_interval == 0:
        tracker.log_eval(
            iteration=iter_num,
            train_loss=train_loss,
            val_loss=val_loss,
        )
```

### Step 4: Finish and Save

```python
tracker.finish(
    train_loss=final_train_loss,
    val_loss=final_val_loss,
    notes="Description of this experiment"
)
tracker.save()
tracker.print_summary()
```

## Integration with training_together.py

Here's how to integrate the logger with your existing training script:

```python
# At the top of training_together.py
from cs336_basics.experiment_log import ExperimentTracker, ExperimentConfig

def train(args):
    """Main training function."""
    # ... existing setup code ...
    
    # After model and optimizer initialization, create tracker
    config = ExperimentConfig(
        experiment_name=args.wandb_run_name or "default_experiment",
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        max_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        seed=args.seed,
        device=args.device,
        optimizer_name="AdamW",
    )
    
    tracker = ExperimentTracker(config, log_dir="experiment_logs")
    tracker.start()
    
    # ... existing training setup ...
    
    # In the training loop
    for iter_num in range(start_iter, args.max_iters):
        iter_start_time = time.time()
        
        # ... existing training code ...
        
        # Log metrics
        if (iter_num + 1) % args.log_interval == 0:
            avg_loss = np.mean(train_losses[-args.log_interval:])
            avg_time = np.mean(iter_times[-args.log_interval:])
            tokens_per_sec = (args.batch_size * args.context_length) / avg_time
            
            tracker.log_metrics(
                iteration=iter_num,
                train_loss=avg_loss,
                learning_rate=lr,
                tokens_per_sec=tokens_per_sec,
                iter_time_ms=avg_time * 1000,
            )
            tracker.increment_gradient_step()
            
            # ... existing logging code ...
        
        # Validation logging
        if (iter_num + 1) % args.eval_interval == 0:
            losses = estimate_loss(...)  # existing code
            
            tracker.log_eval(
                iteration=iter_num,
                train_loss=losses['train'],
                val_loss=losses['val'],
            )
    
    # After training loop
    tracker.finish(
        train_loss=losses['train'],
        val_loss=losses['val'],
        notes="Training completed successfully"
    )
    tracker.save()
    tracker.print_summary()
```

## Analyzing Results

### Load and Compare Experiments

```python
from cs336_basics.experiment_analysis import ExperimentAnalyzer

# Load all experiments
analyzer = ExperimentAnalyzer(log_dir="experiment_logs")
analyzer.load_all()

# Print comparison table
print(analyzer.get_comparison_table())

# Get details for specific experiment
analyzer.print_summary("my_experiment")

# Find best experiment
best_exp, best_loss = analyzer.get_best_experiment("val_loss")
print(f"Best experiment: {best_exp} with loss {best_loss:.6f}")

# Export to CSV for plotting
analyzer.export_csv("my_experiment", Path("my_experiment.csv"))
```

### Quick Summary

```python
from cs336_basics.experiment_analysis import print_experiment_summary

print_experiment_summary()
```

## Metrics Tracked

### Per-Iteration Metrics

- **iteration**: Iteration number (0-indexed)
- **gradient_step**: Total gradient steps since training start
- **wallclock_time**: Seconds elapsed since training start
- **train_loss**: Training loss
- **learning_rate**: Current learning rate
- **tokens_per_sec**: Throughput in tokens per second
- **iter_time_ms**: Time for this iteration in milliseconds

### Validation Metrics

- **train_loss**: Average training loss during validation
- **val_loss**: Average validation loss

### Summary Statistics

Computed automatically after training:

- **best_train_loss**: Minimum training loss achieved
- **best_val_loss**: Minimum validation loss achieved
- **final_train_loss**: Training loss at training completion
- **final_val_loss**: Validation loss at training completion
- **duration_seconds**: Total training time

## Output Files

### JSON Logs

Location: `experiment_logs/experiment_<name>_<timestamp>.json`

Each JSON file contains:

```json
{
  "config": {
    "experiment_name": "my_experiment",
    "d_model": 512,
    "num_heads": 8,
    ...
  },
  "start_time": "2025-02-22T14:05:30.123456",
  "end_time": "2025-02-22T14:15:45.654321",
  "duration_seconds": 615.5,
  "best_train_loss": 2.1234,
  "best_val_loss": 2.3456,
  "final_train_loss": 2.0987,
  "final_val_loss": 2.3210,
  "metrics_history": [
    {
      "iteration": 0,
      "gradient_step": 1,
      "wallclock_time": 0.234,
      "train_loss": 4.5678,
      "val_loss": null,
      "learning_rate": 1.2e-05,
      "tokens_per_sec": 1024.5,
      "iter_time_ms": 2.34
    },
    ...
  ],
  "notes": "Baseline experiment with default hyperparameters"
}
```

### CSV Export

Location: `experiment_logs/<experiment_name>_metrics.csv`

Columns: iteration, gradient_step, wallclock_time, train_loss, val_loss, learning_rate, tokens_per_sec, iter_time_ms

## Tips and Best Practices

### 1. Naming Conventions

Use descriptive experiment names:

```python
# Good
config.experiment_name = "baseline_tinystories"
config.experiment_name = "lr_sweep_1e-4"
config.experiment_name = "double_capacity_512"

# Avoid
config.experiment_name = "exp1"
config.experiment_name = "test"
```

### 2. Recording Notes

Always record what makes each experiment different:

```python
tracker.finish(
    notes="Doubled d_model from 256 to 512. "
          "Expectation: better capacity but slower convergence."
)
```

### 3. Visualizing Loss Curves

Use the CSV export for plotting with your favorite tool:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("my_experiment_metrics.csv")
plt.plot(df['iteration'], df['train_loss'], label='Train')
plt.plot(df['iteration'], df['val_loss'], label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### 4. Batch Processing

Compare results across multiple experiments:

```python
analyzer = ExperimentAnalyzer()
analyzer.load_all()

for exp_name in analyzer.logs:
    analyzer.print_summary(exp_name)
    analyzer.export_csv(exp_name, Path(f"results/{exp_name}.csv"))
```

### 5. Tracking Gradient Steps vs Iterations

- **Iteration**: Index in your training loop
- **Gradient Step**: Every time you call `optimizer.step()`

Usually they're the same, but if using gradient accumulation, they differ:

```python
# Without accumulation (same)
tracker.increment_gradient_step()  # Every iteration

# With accumulation (different)
if (iter_num + 1) % accumulation_steps == 0:
    tracker.increment_gradient_step()  # Only when optimizer steps
```

## Common Patterns

### Pattern 1: Quick Baseline

```python
config = ExperimentConfig(experiment_name="quick_baseline")
tracker = ExperimentTracker(config)
tracker.start()
# ... training ...
tracker.finish()
tracker.save()
tracker.print_summary()
```

### Pattern 2: Hyperparameter Sweep

```python
for lr in [1e-4, 6e-4, 1e-3]:
    for batch_size in [32, 64, 128]:
        config = ExperimentConfig(
            experiment_name=f"lr_{lr:.0e}_bs_{batch_size}",
            max_learning_rate=lr,
            batch_size=batch_size,
        )
        tracker = ExperimentTracker(config)
        tracker.start()
        # ... training ...
        tracker.finish()
        tracker.save()

# Compare all
analyzer = ExperimentAnalyzer()
analyzer.load_all()
print(analyzer.get_comparison_table())
```

### Pattern 3: Ablation Study

```python
base_config = {
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
}

ablations = [
    {"d_model": 256},  # Half model size
    {"num_heads": 4},  # Half heads
    {"num_layers": 3}, # Half layers
]

for ablation in ablations:
    params = {**base_config, **ablation}
    config = ExperimentConfig(
        experiment_name=f"ablation_{'_'.join(f'{k}_{v}' for k,v in ablation.items())}",
        **params
    )
    tracker = ExperimentTracker(config)
    # ... training ...
```

## Troubleshooting

### Issue: Logs not being saved

**Solution**: Ensure `experiment_logs/` directory exists and is writable:

```python
Path("experiment_logs").mkdir(parents=True, exist_ok=True)
```

### Issue: Want to resume tracking from checkpoint

**Solution**: You can load the previous log:

```python
from cs336_basics.experiment_log import load_experiment_log

# Load previous log
old_log = load_experiment_log(Path("experiment_logs/previous_experiment.json"))

# Create new tracker with same config
tracker = ExperimentTracker(old_log.config)
tracker.start()
# ... continue training ...
```

### Issue: Metrics history too large

**Solution**: The logs can get large with many iterations. You can:

1. Sample metrics (log every Nth iteration)
2. Compress and archive old logs
3. Use CSV export for long-term storage

## Reference

For more details, see:

- [experiment_log.py](./cs336_basics/experiment_log.py) - Implementation
- [experiment_analysis.py](./cs336_basics/experiment_analysis.py) - Analysis tools
- [experiment_tracking_example.py](./cs336_basics/experiment_tracking_example.py) - Usage examples
- [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) - Experiment documentation
