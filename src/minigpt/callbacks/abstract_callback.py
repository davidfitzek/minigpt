from typing import Any


class Callback:
    """Base callback class for trainer events."""

    def on_fit_begin(self, trainer: Any) -> None:
        """Called when training begins."""
        pass

    def on_fit_end(self, trainer: Any) -> None:
        """Called when training ends."""
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, trainer: Any, step: int) -> None:
        """Called at the beginning of each step."""
        pass

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        """Called at the end of each step."""
        pass

    def on_evaluate(self, trainer: Any, train_loss: float, val_loss: float) -> None:
        """Called after evaluation."""
        pass
