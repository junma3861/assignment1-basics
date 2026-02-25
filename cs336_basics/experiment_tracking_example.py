"""
Example script demonstrating how to use the experiment tracking infrastructure.

This example shows:
1. How to set up an ExperimentTracker
2. How to log metrics during training
3. How to save and analyze results
4. How to compare multiple experiments

Run this example with:
    python cs336_basics/experiment_tracking_example.py
"""

import torch
import torch.nn as nn
from pathlib import Path
from cs336_basics.experiment_log import ExperimentTracker, ExperimentConfig
from cs336_basics.experiment_analysis import ExperimentAnalyzer


def create_simple_model(vocab_size: int, d_model: int) -> nn.Module:
    """Create a simple model for demonstration."""
    return nn.Sequential(
        nn.Embedding(vocab_size, d_model),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, vocab_size),
    )


def example_basic_logging():
    """
    Example 1: Basic logging and tracking.
    
    This demonstrates the minimal setup needed to track an experiment.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Experiment Tracking")
    print("=" * 80)
    
    # Step 1: Create configuration
    config = ExperimentConfig(
        experiment_name="example_basic",
        vocab_size=100,
        context_length=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
        batch_size=16,
        max_iters=100,
        max_learning_rate=1e-3,
        warmup_iters=10,
    )
    print(f"\n✓ Created config for '{config.experiment_name}'")
    
    # Step 2: Initialize tracker
    tracker = ExperimentTracker(config, log_dir="experiment_logs")
    tracker.start()
    print("✓ Tracker initialized and started")
    
    # Step 3: Simulate training loop
    model = create_simple_model(config.vocab_size, config.d_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.max_learning_rate)
    
    for iter_num in range(config.max_iters):
        # Simulate training step
        inputs = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
        targets = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
        
        logits = model[:-1](inputs)  # All but last layer
        loss = nn.functional.cross_entropy(
            model[-1](logits.view(-1, config.d_model)),
            targets.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Increment gradient step counter
        tracker.increment_gradient_step()
        
        # Log training metrics
        lr = config.max_learning_rate * (1 - iter_num / config.max_iters)  # Simple schedule
        tracker.log_metrics(
            iteration=iter_num,
            train_loss=loss.item(),
            learning_rate=lr,
            tokens_per_sec=100.0,  # Mock value
            iter_time_ms=1.0,  # Mock value
        )
        
        # Simulate periodic validation
        if (iter_num + 1) % 25 == 0:
            val_loss = loss.item() * 1.1  # Mock: validation loss slightly higher
            tracker.log_eval(
                iteration=iter_num,
                train_loss=loss.item(),
                val_loss=val_loss,
            )
            print(f"  Iteration {iter_num + 1}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
    
    # Step 4: Finish tracking
    tracker.finish(
        train_loss=loss.item(),
        val_loss=val_loss,
        notes="Example basic tracking - synthetic data"
    )
    print("✓ Training finished")
    
    # Step 5: Save and print summary
    log_path = tracker.save()
    print(f"✓ Log saved to {log_path}")
    
    tracker.print_summary()
    
    return log_path


def example_multiple_runs():
    """
    Example 2: Multiple experiments with different configurations.
    
    This demonstrates how to run multiple experiments and compare them.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Experiments (Quick Demo)")
    print("=" * 80)
    
    # Run a few quick experiments with different learning rates
    learning_rates = [1e-3, 5e-4, 1e-4]
    log_paths = []
    
    for lr in learning_rates:
        config = ExperimentConfig(
            experiment_name=f"example_lr_{lr:.0e}",
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_heads=4,
            num_layers=2,
            batch_size=16,
            max_iters=50,  # Quick iteration
            max_learning_rate=lr,
            warmup_iters=5,
        )
        
        tracker = ExperimentTracker(config)
        tracker.start()
        
        # Quick training simulation
        model = create_simple_model(config.vocab_size, config.d_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for iter_num in range(config.max_iters):
            inputs = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
            targets = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
            
            logits = model[:-1](inputs)
            loss = nn.functional.cross_entropy(
                model[-1](logits.view(-1, config.d_model)),
                targets.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tracker.increment_gradient_step()
            
            tracker.log_metrics(
                iteration=iter_num,
                train_loss=loss.item(),
                learning_rate=lr,
            )
            
            if (iter_num + 1) % 10 == 0:
                val_loss = loss.item() * 1.05
                tracker.log_eval(iteration=iter_num, train_loss=loss.item(), val_loss=val_loss)
        
        tracker.finish(train_loss=loss.item(), val_loss=val_loss, notes=f"LR={lr:.0e}")
        log_paths.append(tracker.save())
        print(f"  ✓ Completed experiment with LR={lr:.0e}")
    
    return log_paths


def example_analysis():
    """
    Example 3: Analyzing and comparing experiments.
    
    This demonstrates how to load and compare multiple experiment logs.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Experiment Analysis and Comparison")
    print("=" * 80)
    
    analyzer = ExperimentAnalyzer(log_dir="experiment_logs")
    analyzer.load_all()
    
    if analyzer.logs:
        print("\n" + analyzer.get_comparison_table())
        
        # Print detailed summary for best experiment
        best = analyzer.get_best_experiment("val_loss")
        if best:
            print(f"\n✓ Best experiment by validation loss: {best[0]}")
            analyzer.print_summary(best[0])
        
        # Export to CSV for further analysis
        for exp_name in analyzer.logs.keys():
            csv_path = Path("experiment_logs") / f"{exp_name}_metrics.csv"
            analyzer.export_csv(exp_name, csv_path)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("EXPERIMENT TRACKING INFRASTRUCTURE - EXAMPLES")
    print("=" * 80)
    
    # Example 1: Basic tracking
    example_basic_logging()
    
    # Example 2: Multiple runs
    example_multiple_runs()
    
    # Example 3: Analysis
    example_analysis()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey takeaways:")
    print("1. ExperimentTracker is initialized with an ExperimentConfig")
    print("2. Call tracker.start() to begin, tracker.finish() when done")
    print("3. Log metrics with tracker.log_metrics() during training")
    print("4. Log validation results with tracker.log_eval()")
    print("5. Save with tracker.save() and analyze with ExperimentAnalyzer")
    print("\nLogs are saved to: experiment_logs/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
