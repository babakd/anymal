"""Training utilities.

Heavy trainer imports are lazy so scalar-only helpers such as the health monitor
can be tested without importing torch.
"""

__all__ = [
    "Trainer",
    "PretrainTrainer",
    "FinetuneTrainer",
    "setup_distributed",
    "cleanup_distributed",
    "TrainingHealthMonitor",
    "HealthMonitorConfig",
    "ThroughputTracker",
]


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    if name == "PretrainTrainer":
        from .pretrain import PretrainTrainer

        return PretrainTrainer
    if name == "FinetuneTrainer":
        from .finetune import FinetuneTrainer

        return FinetuneTrainer
    if name in {"setup_distributed", "cleanup_distributed"}:
        from .distributed import cleanup_distributed, setup_distributed

        return {
            "setup_distributed": setup_distributed,
            "cleanup_distributed": cleanup_distributed,
        }[name]
    if name in {"TrainingHealthMonitor", "HealthMonitorConfig"}:
        from .health_monitor import HealthMonitorConfig, TrainingHealthMonitor

        return {
            "TrainingHealthMonitor": TrainingHealthMonitor,
            "HealthMonitorConfig": HealthMonitorConfig,
        }[name]
    if name == "ThroughputTracker":
        from .throughput_tracker import ThroughputTracker

        return ThroughputTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
