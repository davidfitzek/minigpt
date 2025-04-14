import torch
from datetime import datetime
from typing import Dict, Optional, Any, Union
import wandb
from minigpt.loggers.abstract_logger import Logger


class WandbLogger(Logger):
    """Weights & Biases logger."""

    def __init__(
        self,
        project: str = "gpt-training",
        name: Optional[str] = None,
        log_model: Union[str, bool] = False,
        **kwargs,
    ) -> None:
        """
        Initialize WandbLogger.

        Args:
            project: W&B project name
            name: W&B run name
            log_model: If "all", upload all checkpoints. If True, upload best checkpoint only.
            **kwargs: Additional arguments to pass to wandb.init
        """

        self.project = project
        self.name = name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.log_model = log_model
        self.kwargs = kwargs

        # Initialize wandb
        wandb.init(project=self.project, name=self.name, **self.kwargs)
        self.experiment = wandb

    def watch(
        self, model: torch.nn.Module, log: str = "gradients", log_freq: int = 100
    ) -> None:
        """Watch model parameters and gradients."""
        self.experiment.watch(model, log=log, log_freq=log_freq)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        self.experiment.log(metrics, step=step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to wandb."""
        self.experiment.config.update(params)

    def save(self) -> None:
        """Save wandb data."""
        # wandb handles saving automatically
        pass

    def finalize(self) -> None:
        """Clean up after training completes."""
        self.experiment.finish()
