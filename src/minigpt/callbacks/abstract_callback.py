from typing import Any

# NOTE Abstract class to build several new callbacks such as Timer, EarlyStopping, etc.


class Callback:
    """Base callback class for trainer events."""

    def on_fit_start(self, trainer: Any) -> None:
        """Called when training begins."""
        pass

    def on_fit_end(self, trainer: Any) -> None:
        """Called when training ends."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        """Called at the end of each batch."""
        pass

    def on_validation_start(self, trainer: Any) -> None:
        """Called at the start of validation."""
        pass

    def on_validation_end(
        self, trainer: Any, train_loss: float, val_loss: float
    ) -> None:
        """Called at the end of validation."""
        pass
