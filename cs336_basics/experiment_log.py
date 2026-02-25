"""
Experiment tracking and logging infrastructure for monitoring training runs.

This module provides tools to log and track training experiments with support for:
- Real-time metric tracking (loss, learning rate, throughput)
- Wallclock time monitoring
- Gradient step counting
- JSON-based experiment logs for reproducibility
- Easy experiment comparison and analysis
"""

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class MetricsSnapshot:
    """A single snapshot of training metrics at a specific point in time."""
    iteration: int
    gradient_step: int
    wallclock_time: float  # seconds elapsed since training start
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    tokens_per_sec: float = 0.0
    iter_time_ms: float = 0.0
    batch_size: int = 0
    context_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_name: str
    model_name: str = "TransformerLM"
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    num_layers: int = 6
    rope_theta: float = 10000.0
    batch_size: int = 64
    max_iters: int = 100000
    max_learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_iters: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    seed: int = 42
    device: str = "cuda"
    optimizer_name: str = "AdamW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ExperimentLog:
    """Complete log of an experiment run."""
    config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    metrics_history: List[MetricsSnapshot] = field(default_factory=list)
    notes: str = ""
    best_train_loss: float = float('inf')
    best_val_loss: float = float('inf')
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time is not None:
            data['end_time'] = self.end_time.isoformat()
        data['metrics_history'] = [m.to_dict() for m in self.metrics_history]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentLog':
        """Reconstruct from dictionary."""
        data = data.copy()
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time') is not None:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        config_data = data.pop('config')
        data['config'] = ExperimentConfig(**config_data)
        metrics_data = data.pop('metrics_history', [])
        data['metrics_history'] = [MetricsSnapshot(**m) for m in metrics_data]
        return cls(**data)


class ExperimentTracker:
    """
    Tracks and logs training metrics for an experiment.
    
    Usage:
        tracker = ExperimentTracker(config, log_dir="logs")
        tracker.start()
        
        for step in range(max_steps):
            # ... training step ...
            tracker.log_metrics(
                iteration=step,
                train_loss=loss.item(),
                learning_rate=current_lr
            )
        
        tracker.log_eval(val_loss=val_loss)
        tracker.finish()
        tracker.save()
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        log_dir: str = "experiment_logs"
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            config: ExperimentConfig with model and training parameters
            log_dir: Directory to save experiment logs
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log = ExperimentLog(
            config=config,
            start_time=datetime.now()
        )
        
        self.start_wall_time: Optional[float] = None
        self.gradient_step_count = 0
        self.last_log_time = None
    
    def start(self) -> None:
        """Mark the start of training and record wallclock time."""
        self.start_wall_time = time.time()
        print(f"Experiment '{self.config.experiment_name}' started at {self.log.start_time.isoformat()}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since start."""
        if self.start_wall_time is None:
            return 0.0
        return time.time() - self.start_wall_time
    
    def increment_gradient_step(self) -> None:
        """Increment the gradient step counter."""
        self.gradient_step_count += 1
    
    def log_metrics(
        self,
        iteration: int,
        train_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        tokens_per_sec: float = 0.0,
        iter_time_ms: float = 0.0,
    ) -> None:
        """
        Log training metrics for a specific iteration.
        
        Args:
            iteration: Iteration number (0-indexed)
            train_loss: Training loss value
            learning_rate: Current learning rate
            tokens_per_sec: Throughput in tokens per second
            iter_time_ms: Time for this iteration in milliseconds
        """
        snapshot = MetricsSnapshot(
            iteration=iteration,
            gradient_step=self.gradient_step_count,
            wallclock_time=self.get_elapsed_time(),
            train_loss=train_loss,
            learning_rate=learning_rate,
            tokens_per_sec=tokens_per_sec,
            iter_time_ms=iter_time_ms,
            batch_size=self.config.batch_size,
            context_length=self.config.context_length,
        )
        
        self.log.metrics_history.append(snapshot)
        
        if train_loss is not None and train_loss < self.log.best_train_loss:
            self.log.best_train_loss = train_loss
    
    def log_eval(
        self,
        iteration: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> None:
        """
        Log validation metrics.
        
        Args:
            iteration: Iteration at which evaluation occurred
            train_loss: Training loss at evaluation time
            val_loss: Validation loss at evaluation time
        """
        # Find the corresponding metric snapshot or create a new one
        if self.log.metrics_history and self.log.metrics_history[-1].iteration == iteration:
            # Update the last snapshot with validation loss
            self.log.metrics_history[-1].val_loss = val_loss
        else:
            # Create a new snapshot specifically for evaluation
            snapshot = MetricsSnapshot(
                iteration=iteration,
                gradient_step=self.gradient_step_count,
                wallclock_time=self.get_elapsed_time(),
                train_loss=train_loss,
                val_loss=val_loss,
                batch_size=self.config.batch_size,
                context_length=self.config.context_length,
            )
            self.log.metrics_history.append(snapshot)
        
        if val_loss is not None and val_loss < self.log.best_val_loss:
            self.log.best_val_loss = val_loss
    
    def finish(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        notes: str = ""
    ) -> None:
        """
        Mark the end of training.
        
        Args:
            train_loss: Final training loss
            val_loss: Final validation loss
            notes: Additional notes about the experiment
        """
        self.log.end_time = datetime.now()
        self.log.duration_seconds = self.get_elapsed_time()
        self.log.final_train_loss = train_loss
        self.log.final_val_loss = val_loss
        self.log.notes = notes
        
        print(f"\nExperiment '{self.config.experiment_name}' finished.")
        print(f"Duration: {self._format_duration(self.log.duration_seconds)}")
        if self.log.best_val_loss < float('inf'):
            print(f"Best val loss: {self.log.best_val_loss:.4f}")
    
    def save(self) -> Path:
        """
        Save the experiment log to a JSON file.
        
        Returns:
            Path to the saved log file
        """
        # Create filename with timestamp
        timestamp = self.log.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{self.config.experiment_name}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(self.log.to_dict(), f, indent=2)
        
        print(f"Experiment log saved to: {filepath}")
        return filepath
    
    def load_from_file(self, filepath: Path) -> None:
        """Load experiment log from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.log = ExperimentLog.from_dict(data)
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the experiment.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.log.metrics_history:
            return {}
        
        metrics = self.log.metrics_history
        train_losses = [m.train_loss for m in metrics if m.train_loss is not None]
        val_losses = [m.val_loss for m in metrics if m.val_loss is not None]
        iter_times = [m.iter_time_ms for m in metrics if m.iter_time_ms > 0]
        throughputs = [m.tokens_per_sec for m in metrics if m.tokens_per_sec > 0]
        
        stats = {
            "total_iterations": max([m.iteration for m in metrics], default=0),
            "total_gradient_steps": max([m.gradient_step for m in metrics], default=0),
            "total_duration_seconds": self.log.duration_seconds,
            "best_train_loss": self.log.best_train_loss,
            "best_val_loss": self.log.best_val_loss,
        }
        
        if train_losses:
            stats["train_loss_stats"] = {
                "min": float(np.min(train_losses)),
                "max": float(np.max(train_losses)),
                "mean": float(np.mean(train_losses)),
                "std": float(np.std(train_losses)),
            }
        
        if val_losses:
            stats["val_loss_stats"] = {
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses)),
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
            }
        
        if iter_times:
            stats["iter_time_stats_ms"] = {
                "mean": float(np.mean(iter_times)),
                "std": float(np.std(iter_times)),
            }
        
        if throughputs:
            stats["throughput_stats_tokens_per_sec"] = {
                "mean": float(np.mean(throughputs)),
                "max": float(np.max(throughputs)),
            }
        
        return stats
    
    def print_summary(self) -> None:
        """Print a summary of the experiment to stdout."""
        print("\n" + "=" * 80)
        print(f"EXPERIMENT SUMMARY: {self.config.experiment_name}")
        print("=" * 80)
        
        stats = self.get_summary_stats()
        
        if "total_iterations" in stats:
            print(f"Total iterations: {stats['total_iterations']}")
        if "total_duration_seconds" in stats:
            print(f"Total duration: {self._format_duration(stats['total_duration_seconds'])}")
        
        if "best_train_loss" in stats and stats["best_train_loss"] < float('inf'):
            print(f"Best training loss: {stats['best_train_loss']:.6f}")
        if "best_val_loss" in stats and stats["best_val_loss"] < float('inf'):
            print(f"Best validation loss: {stats['best_val_loss']:.6f}")
        
        if "train_loss_stats" in stats:
            tls = stats["train_loss_stats"]
            print(f"Training loss (final {len(self.log.metrics_history)} measurements): "
                  f"min={tls['min']:.6f}, max={tls['max']:.6f}, mean={tls['mean']:.6f}")
        
        if "val_loss_stats" in stats:
            vls = stats["val_loss_stats"]
            print(f"Validation loss: "
                  f"min={vls['min']:.6f}, max={vls['max']:.6f}, mean={vls['mean']:.6f}")
        
        if "iter_time_stats_ms" in stats:
            its = stats["iter_time_stats_ms"]
            print(f"Iteration time: mean={its['mean']:.2f}ms, std={its['std']:.2f}ms")
        
        if "throughput_stats_tokens_per_sec" in stats:
            tps = stats["throughput_stats_tokens_per_sec"]
            print(f"Throughput: mean={tps['mean']:.0f} tokens/sec, "
                  f"max={tps['max']:.0f} tokens/sec")
        
        if self.log.notes:
            print(f"\nNotes: {self.log.notes}")
        
        print("=" * 80 + "\n")


def load_experiment_log(filepath: Path) -> ExperimentLog:
    """
    Load an experiment log from a JSON file.
    
    Args:
        filepath: Path to the experiment log JSON file
        
    Returns:
        ExperimentLog object
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ExperimentLog.from_dict(data)


def compare_experiments(log_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load and compare multiple experiment logs.
    
    Args:
        log_files: List of paths to experiment log JSON files
        
    Returns:
        Dictionary mapping experiment names to their summary statistics
    """
    results = {}
    for log_file in log_files:
        log = load_experiment_log(log_file)
        # Calculate summary stats
        metrics = log.metrics_history
        if metrics:
            train_losses = [m.train_loss for m in metrics if m.train_loss is not None]
            val_losses = [m.val_loss for m in metrics if m.val_loss is not None]
            
            results[log.config.experiment_name] = {
                "best_train_loss": log.best_train_loss,
                "best_val_loss": log.best_val_loss,
                "final_train_loss": log.final_train_loss,
                "final_val_loss": log.final_val_loss,
                "duration_seconds": log.duration_seconds,
                "num_metrics": len(metrics),
                "num_val_evals": len(val_losses),
            }
    
    return results
