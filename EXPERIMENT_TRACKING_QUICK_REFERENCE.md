# Experiment Tracking - Quick Reference

## Basic Setup (Copy-Paste Ready)

### Import
```python
from cs336_basics.experiment_log import ExperimentTracker, ExperimentConfig
from cs336_basics.experiment_analysis import ExperimentAnalyzer
```

### Before Training Loop
```python
config = ExperimentConfig(
    experiment_name="my_experiment",
    d_model=512,
    num_heads=8,
    num_layers=6,
    batch_size=64,
    max_iters=100000,
    max_learning_rate=6e-4,
    min_learning_rate=6e-5,
    warmup_iters=2000,
)

tracker = ExperimentTracker(config, log_dir="experiment_logs")
tracker.start()
```

### Inside Training Loop (every iteration or every N iterations)
```python
tracker.log_metrics(
    iteration=iter_num,
    train_loss=loss.item(),
    learning_rate=lr,
    tokens_per_sec=batch_size * context_length / iter_time,
    iter_time_ms=iter_time * 1000,
)
tracker.increment_gradient_step()
```

### During Validation (every N iterations)
```python
tracker.log_eval(
    iteration=iter_num,
    train_loss=train_loss,
    val_loss=val_loss,
)
```

### After Training Loop
```python
tracker.finish(
    train_loss=final_train_loss,
    val_loss=final_val_loss,
    notes="Description of this experiment"
)
tracker.save()
tracker.print_summary()
```

---

## Common Tasks

### Task: Compare all experiments
```python
analyzer = ExperimentAnalyzer()
analyzer.load_all()
print(analyzer.get_comparison_table())
```

### Task: Print details of one experiment
```python
analyzer.print_summary("experiment_name")
```

### Task: Find best experiment by validation loss
```python
best_name, best_loss = analyzer.get_best_experiment("val_loss")
print(f"Best: {best_name} ({best_loss:.6f})")
```

### Task: Export to CSV for plotting
```python
analyzer.export_csv("experiment_name", Path("results.csv"))
```

### Task: Get loss trajectory data
```python
iterations, train_losses, val_losses = analyzer.get_loss_trajectories("experiment_name")
```

### Task: Get throughput data
```python
iterations, tokens_per_sec = analyzer.get_throughput_trajectory("experiment_name")
```

---

## Data Structures

### ExperimentConfig
```python
config = ExperimentConfig(
    # Required
    experiment_name: str,
    
    # Model architecture
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    vocab_size: int = 10000,
    context_length: int = 256,
    rope_theta: float = 10000.0,
    
    # Training
    batch_size: int = 64,
    max_iters: int = 100000,
    max_learning_rate: float = 6e-4,
    min_learning_rate: float = 6e-5,
    warmup_iters: int = 2000,
    
    # Optimizer (AdamW)
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    
    # Other
    seed: int = 42,
    device: str = "cuda",
)
```

### ExperimentTracker Methods
```python
tracker = ExperimentTracker(config, log_dir="experiment_logs")

# Setup and teardown
tracker.start()                  # Begin timing
tracker.finish(...)              # Mark completion
tracker.save()                   # Save to JSON

# Logging
tracker.log_metrics(
    iteration: int,
    train_loss: float = None,
    learning_rate: float = 0.0,
    tokens_per_sec: float = 0.0,
    iter_time_ms: float = 0.0,
)

tracker.log_eval(
    iteration: int,
    train_loss: float = None,
    val_loss: float = None,
)

tracker.increment_gradient_step()  # +1 to gradient step counter

# Utilities
tracker.get_elapsed_time()         # Seconds since start
tracker.get_summary_stats()        # Dict of summary stats
tracker.print_summary()            # Pretty-print summary
```

### ExperimentAnalyzer Methods
```python
analyzer = ExperimentAnalyzer(log_dir="experiment_logs")

# Loading
analyzer.load_all()                        # Load all .json files
analyzer.load_specific(["exp1", "exp2"])   # Load by name

# Display
analyzer.get_comparison_table()            # Formatted string
analyzer.print_summary("exp_name")         # Print one
analyzer.get_best_experiment("val_loss")   # Return (name, value)

# Data extraction
analyzer.get_loss_trajectories("exp_name")      # (iters, train, val)
analyzer.get_throughput_trajectory("exp_name")  # (iters, tokens/sec)
analyzer.get_learning_rate_trajectory("exp_name") # (iters, lrs)

# Export
analyzer.export_csv("exp_name", Path("file.csv"))
```

---

## Metrics Logged

| Metric | Type | Units | Notes |
|--------|------|-------|-------|
| iteration | int | number | Loop index (0-based) |
| gradient_step | int | count | Total optimizer.step() calls |
| wallclock_time | float | seconds | ⏱️ Elapsed time since start |
| train_loss | float | loss | Training loss value |
| val_loss | float | loss | Validation loss value |
| learning_rate | float | LR | Current learning rate |
| tokens_per_sec | float | tokens/sec | 📊 Throughput (compute efficiency) |
| iter_time_ms | float | milliseconds | ⏱️ Time for this iteration |

---

## Output Files

All logs saved to: `experiment_logs/`

Filename: `experiment_<name>_<timestamp>.json`

Example: `experiment_baseline_tinystories_20250222_140530.json`

Each file contains:
- **config**: All hyperparameters (for reproducibility)
- **start_time, end_time**: ISO format timestamps
- **duration_seconds**: Total training time
- **metrics_history**: Array of MetricsSnapshot objects
- **best_train_loss, best_val_loss**: Best values found
- **notes**: Your notes about this experiment

---

## Integration Snippet for training_together.py

Add to imports:
```python
from cs336_basics.experiment_log import ExperimentTracker, ExperimentConfig
```

After model initialization:
```python
config = ExperimentConfig(
    experiment_name=args.wandb_run_name or "default",
    d_model=args.d_model,
    num_heads=args.num_heads,
    # ... other args ...
)
tracker = ExperimentTracker(config)
tracker.start()
```

In training loop (after loss backward/step):
```python
if (iter_num + 1) % args.log_interval == 0:
    tracker.log_metrics(
        iteration=iter_num,
        train_loss=avg_loss,
        learning_rate=lr,
        tokens_per_sec=tokens_per_sec,
        iter_time_ms=avg_time * 1000,
    )
    tracker.increment_gradient_step()
```

During validation:
```python
tracker.log_eval(
    iteration=iter_num,
    train_loss=losses['train'],
    val_loss=losses['val'],
)
```

After training loop:
```python
tracker.finish(train_loss=losses['train'], val_loss=losses['val'])
tracker.save()
tracker.print_summary()
```

---

## Example Configurations

### Baseline
```python
ExperimentConfig(
    experiment_name="baseline",
    d_model=512, num_heads=8, num_layers=6,
    batch_size=64, max_iters=10000,
)
```

### Larger Model
```python
ExperimentConfig(
    experiment_name="large",
    d_model=768, num_heads=12, num_layers=8,
    batch_size=64, max_iters=10000,
)
```

### Learning Rate Sweep
```python
for lr in [1e-4, 6e-4, 1e-3]:
    ExperimentConfig(
        experiment_name=f"lr_{lr:.0e}",
        max_learning_rate=lr,
        # ... other params ...
    )
```

### Batch Size Study
```python
for bs in [32, 64, 128]:
    ExperimentConfig(
        experiment_name=f"batch_{bs}",
        batch_size=bs,
        # ... other params ...
    )
```

---

## Troubleshooting Checklist

- [ ] Logs directory exists: `mkdir experiment_logs`
- [ ] `tracker.start()` called before training loop
- [ ] `tracker.log_metrics()` called during loop
- [ ] `tracker.increment_gradient_step()` called every iteration
- [ ] `tracker.finish()` called after training
- [ ] `tracker.save()` called to write JSON
- [ ] Config experiment_name is unique/descriptive
- [ ] Loading logs: ensure .json files in `experiment_logs/`

---

## Files to Reference

- **Integration in your code**: `EXPERIMENT_TRACKING_GUIDE.md`
- **Experiment documentation**: `EXPERIMENT_LOG.md` (edit this)
- **Working examples**: `cs336_basics/experiment_tracking_example.py`
- **Implementation**: `cs336_basics/experiment_log.py`
- **Analysis tools**: `cs336_basics/experiment_analysis.py`

---

## Tips

1. **Descriptive names**: Use names like `baseline`, `lr_1e-3`, `double_capacity`
2. **Record notes**: Document what makes each experiment different
3. **Save early**: Call `tracker.finish()` and `tracker.save()` immediately after training
4. **Compare regularly**: Run `analyzer.load_all()` and `print(analyzer.get_comparison_table())`
5. **Export for plotting**: Use CSV export to plot loss curves in Excel/Jupyter

---

## API at a Glance

```
ExperimentConfig: NamedTuple-like config object
│
ExperimentTracker: Main logging interface
├── start()
├── log_metrics(iteration, train_loss, lr, ...)
├── log_eval(iteration, val_loss)
├── increment_gradient_step()
├── finish(notes="...")
├── save() → Path
└── print_summary()

ExperimentLog: Complete run history (saved as JSON)
│
ExperimentAnalyzer: Load and compare multiple runs
├── load_all()
├── get_comparison_table() → str
├── print_summary(exp_name)
├── get_best_experiment(metric) → (name, value)
├── get_loss_trajectories(exp_name) → (iters, train, val)
├── export_csv(exp_name, path)
└── logs: Dict[exp_name, ExperimentLog]
```
