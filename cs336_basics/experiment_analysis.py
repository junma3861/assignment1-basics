"""
Utilities for analyzing and visualizing experiment logs.

This module provides tools to:
- Load and compare experiment logs
- Generate summary statistics
- Create visualizations of training curves
- Export results to various formats
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from cs336_basics.experiment_log import ExperimentLog, load_experiment_log, MetricsSnapshot


class ExperimentAnalyzer:
    """Analyze and compare multiple experiment logs."""
    
    def __init__(self, log_dir: Path = Path("experiment_logs")):
        """
        Initialize analyzer with log directory.
        
        Args:
            log_dir: Directory containing experiment log files
        """
        self.log_dir = Path(log_dir)
        self.logs: Dict[str, ExperimentLog] = {}
    
    def load_all(self) -> None:
        """Load all experiment logs from log directory."""
        if not self.log_dir.exists():
            print(f"Log directory not found: {self.log_dir}")
            return
        
        for log_file in sorted(self.log_dir.glob("experiment_*.json")):
            try:
                log = load_experiment_log(log_file)
                self.logs[log.config.experiment_name] = log
                print(f"Loaded: {log.config.experiment_name}")
            except Exception as e:
                print(f"Error loading {log_file}: {e}")
    
    def load_specific(self, experiments: List[str]) -> None:
        """
        Load specific experiments by name.
        
        Args:
            experiments: List of experiment names to load
        """
        for log_file in self.log_dir.glob("experiment_*.json"):
            log = load_experiment_log(log_file)
            if log.config.experiment_name in experiments:
                self.logs[log.config.experiment_name] = log
    
    def get_comparison_table(self) -> str:
        """Get a formatted comparison table of all experiments."""
        if not self.logs:
            return "No experiments loaded"
        
        # Header
        lines = ["Experiment Comparison\n"]
        lines.append("-" * 120)
        lines.append(
            f"{'Name':<30} {'Train Loss':<15} {'Val Loss':<15} "
            f"{'Duration':<15} {'Iterations':<12}"
        )
        lines.append("-" * 120)
        
        # Rows
        for exp_name in sorted(self.logs.keys()):
            log = self.logs[exp_name]
            train_loss = f"{log.best_train_loss:.6f}" if log.best_train_loss < float('inf') else "N/A"
            val_loss = f"{log.best_val_loss:.6f}" if log.best_val_loss < float('inf') else "N/A"
            duration = self._format_duration(log.duration_seconds)
            
            if log.metrics_history:
                iterations = log.metrics_history[-1].iteration
            else:
                iterations = 0
            
            lines.append(
                f"{exp_name:<30} {train_loss:<15} {val_loss:<15} "
                f"{duration:<15} {iterations:<12}"
            )
        
        lines.append("-" * 120)
        return "\n".join(lines)
    
    def get_loss_trajectories(self, experiment_name: str) -> Tuple[List[int], List[float], List[float]]:
        """
        Get loss trajectory for an experiment.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Tuple of (iterations, train_losses, val_losses)
        """
        if experiment_name not in self.logs:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        log = self.logs[experiment_name]
        metrics = log.metrics_history
        
        iterations = [m.iteration for m in metrics]
        train_losses = [m.train_loss for m in metrics if m.train_loss is not None]
        val_losses = [m.val_loss for m in metrics if m.val_loss is not None]
        
        return iterations, train_losses, val_losses
    
    def get_throughput_trajectory(self, experiment_name: str) -> Tuple[List[int], List[float]]:
        """
        Get throughput (tokens/sec) trajectory for an experiment.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Tuple of (iterations, tokens_per_sec)
        """
        if experiment_name not in self.logs:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        log = self.logs[experiment_name]
        metrics = log.metrics_history
        
        iterations = [m.iteration for m in metrics if m.tokens_per_sec > 0]
        throughputs = [m.tokens_per_sec for m in metrics if m.tokens_per_sec > 0]
        
        return iterations, throughputs
    
    def get_learning_rate_trajectory(self, experiment_name: str) -> Tuple[List[int], List[float]]:
        """
        Get learning rate trajectory for an experiment.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Tuple of (iterations, learning_rates)
        """
        if experiment_name not in self.logs:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        log = self.logs[experiment_name]
        metrics = log.metrics_history
        
        iterations = [m.iteration for m in metrics]
        lrs = [m.learning_rate for m in metrics]
        
        return iterations, lrs
    
    def print_summary(self, experiment_name: str) -> None:
        """Print summary for a specific experiment."""
        if experiment_name not in self.logs:
            print(f"Experiment '{experiment_name}' not found")
            return
        
        log = self.logs[experiment_name]
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {experiment_name}")
        print("=" * 80)
        
        # Configuration
        print("\nConfiguration:")
        print(f"  Model: {log.config.model_name}")
        print(f"  d_model: {log.config.d_model}, num_heads: {log.config.num_heads}, "
              f"num_layers: {log.config.num_layers}")
        print(f"  Vocab size: {log.config.vocab_size}, Context: {log.config.context_length}")
        print(f"  Batch size: {log.config.batch_size}, Max iterations: {log.config.max_iters}")
        print(f"  LR: {log.config.max_learning_rate:.2e} -> {log.config.min_learning_rate:.2e}, "
              f"Warmup: {log.config.warmup_iters}")
        
        # Results
        print("\nResults:")
        print(f"  Duration: {self._format_duration(log.duration_seconds)}")
        print(f"  Best train loss: {log.best_train_loss:.6f}")
        print(f"  Best val loss: {log.best_val_loss:.6f}")
        if log.final_train_loss is not None:
            print(f"  Final train loss: {log.final_train_loss:.6f}")
        if log.final_val_loss is not None:
            print(f"  Final val loss: {log.final_val_loss:.6f}")
        
        # Throughput
        if log.metrics_history:
            throughputs = [m.tokens_per_sec for m in log.metrics_history if m.tokens_per_sec > 0]
            if throughputs:
                print(f"  Mean throughput: {np.mean(throughputs):.0f} tokens/sec")
                print(f"  Max throughput: {np.max(throughputs):.0f} tokens/sec")
        
        if log.notes:
            print(f"\nNotes: {log.notes}")
        
        print("=" * 80 + "\n")
    
    def export_csv(self, experiment_name: str, output_path: Path) -> None:
        """
        Export experiment metrics to CSV.
        
        Args:
            experiment_name: Name of experiment
            output_path: Path to save CSV file
        """
        if experiment_name not in self.logs:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        log = self.logs[experiment_name]
        metrics = log.metrics_history
        
        with open(output_path, 'w') as f:
            # Header
            f.write("iteration,gradient_step,wallclock_time,train_loss,val_loss,"
                    "learning_rate,tokens_per_sec,iter_time_ms\n")
            
            # Data
            for m in metrics:
                f.write(f"{m.iteration},{m.gradient_step},{m.wallclock_time:.2f},"
                        f"{m.train_loss if m.train_loss else 'NA'},"
                        f"{m.val_loss if m.val_loss else 'NA'},"
                        f"{m.learning_rate:.2e},{m.tokens_per_sec:.2f},"
                        f"{m.iter_time_ms:.2f}\n")
        
        print(f"Exported {experiment_name} to {output_path}")
    
    def get_best_experiment(self, metric: str = "val_loss") -> Optional[Tuple[str, float]]:
        """
        Get the best experiment by a given metric.
        
        Args:
            metric: Metric to compare ('val_loss', 'train_loss', 'duration')
            
        Returns:
            Tuple of (experiment_name, metric_value)
        """
        if not self.logs:
            return None
        
        best_exp = None
        best_value = float('inf') if metric.endswith('loss') else float('inf')
        is_minimization = metric.endswith('loss') or metric == 'duration'
        
        for exp_name, log in self.logs.items():
            if metric == "val_loss":
                value = log.best_val_loss
            elif metric == "train_loss":
                value = log.best_train_loss
            elif metric == "duration":
                value = log.duration_seconds
            else:
                continue
            
            if is_minimization:
                if value < best_value:
                    best_value = value
                    best_exp = exp_name
            else:
                if value > best_value:
                    best_value = value
                    best_exp = exp_name
        
        if best_exp is None:
            return None
        
        return best_exp, best_value
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


def print_experiment_summary() -> None:
    """
    Print summary of all experiments in the log directory.
    Useful as a quick check of all experiments run so far.
    """
    analyzer = ExperimentAnalyzer()
    analyzer.load_all()
    
    if not analyzer.logs:
        print("No experiments found")
        return
    
    print("\n" + analyzer.get_comparison_table())
    
    # Find best by val loss
    best = analyzer.get_best_experiment("val_loss")
    if best:
        print(f"\nBest by validation loss: {best[0]} ({best[1]:.6f})")
    
    # Find best by training time
    best = analyzer.get_best_experiment("duration")
    if best:
        print(f"Fastest: {best[0]} ({best[1]:.1f} seconds)")


if __name__ == "__main__":
    print_experiment_summary()
