import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union


class LightningModule(nn.Module, ABC):
    """
    Abstract base class for PyTorch Lightning-style modules.
    This provides the interface for model training components.
    """

    def __init__(self):
        super().__init__()
        self.logged_metrics = {}

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Execute a single training step

        Args:
            batch: Input batch
            batch_idx: Index of the current batch

        Returns:
            loss: The training loss
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Execute a single validation step

        Args:
            batch: Input batch
            batch_idx: Index of the current batch

        Returns:
            loss: The validation loss
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure and return the optimizer

        Returns:
            optimizer: The configured optimizer
        """
        pass

    def log(
        self,
        name: str,
        value: Union[torch.Tensor, float],
        on_step: bool = False,
        on_epoch: bool = True,
        logger: bool = True,
    ) -> None:
        """
        Log a metric value

        Args:
            name: Name of the metric
            value: Value of the metric
            on_step: Whether to log at step level
            on_epoch: Whether to aggregate and log at epoch level
            logger: Whether to send to the logger
        """
        # Convert tensor to float if needed
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.detach()

        # Store the logged metric
        self.logged_metrics[name] = {
            "value": value,
            "on_step": on_step,
            "on_epoch": on_epoch,
            "logger": logger,
        }

    def get_logged_metrics(self) -> Dict:
        """Get all logged metrics for the current step"""
        return self.logged_metrics

    def reset_logged_metrics(self) -> None:
        """Reset logged metrics after they've been processed"""
        self.logged_metrics = {}
