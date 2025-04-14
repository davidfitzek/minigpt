from typing import Dict, Optional, Any
from abc import ABC, abstractmethod


class Logger(ABC):
    """Base class for all loggers."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save logger data."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Clean up after training completes."""
        pass
