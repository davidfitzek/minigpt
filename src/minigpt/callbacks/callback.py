from typing import Any, Optional
import torch
import os
from minigpt.loggers import WandbLogger
from minigpt.callbacks.abstract_callback import Callback


class ModelCheckpoint(Callback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "model_epoch_{epoch}",
        save_freq: str = "epoch",  # "epoch" or "step"
        every_n_epochs: int = 1,
        every_n_steps: Optional[int] = None,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.save_freq = save_freq
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.verbose = verbose

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.dirpath, exist_ok=True)

        # Initialize best value
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def _save_checkpoint(self, trainer: Any, epoch: int) -> None:
        """Save model checkpoint."""
        # Create checkpoint filename with epoch number
        filepath = os.path.join(self.dirpath, self.filename.format(epoch=epoch + 1))

        # Add file extension if not present
        if not filepath.endswith(".pth"):
            filepath += ".pth"

        # Save model state dict
        torch.save(trainer.model.state_dict(), filepath)

        # Log model artifact if using wandb
        if isinstance(trainer.logger, WandbLogger) and trainer.logger.log_model:
            trainer.logger.experiment.save(filepath)

        if self.verbose:
            print(f"Saved checkpoint: {filepath}")

    def on_epoch_end(self, trainer: Any, epoch: int) -> None:
        """Save model at the end of epoch if conditions are met."""
        if self.save_freq == "epoch" and (epoch + 1) % self.every_n_epochs == 0:
            if not self.save_best_only:
                self._save_checkpoint(trainer, epoch)
            elif hasattr(trainer, "val_losses") and trainer.val_losses:
                # Get current value to monitor
                current = (
                    trainer.val_losses[-1]
                    if self.monitor == "val_loss"
                    else trainer.train_losses[-1]
                )

                # Check if current value is better than best value
                is_better = (self.mode == "min" and current < self.best_value) or (
                    self.mode == "max" and current > self.best_value
                )

                if is_better:
                    if self.verbose:
                        print(
                            f"Val loss improved from {self.best_value:.4f} to {current:.4f}"
                        )
                    self.best_value = current
                    self._save_checkpoint(trainer, epoch)

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        """Save model at the end of step if conditions are met."""
        if self.save_freq == "step" and self.every_n_steps is not None:
            if step % self.every_n_steps == 0:
                # Calculate current epoch
                epoch = trainer.global_step // len(trainer.train_loader)
                self._save_checkpoint(trainer, epoch)

    def on_fit_end(self, trainer: Any) -> None:
        """Save last model checkpoint if save_last is True."""
        if self.save_last:
            # Calculate current epoch
            # epoch = trainer.global_step // len(trainer.train_loader)

            # Create last checkpoint filename
            filepath = os.path.join(self.dirpath, "model_last.pth")

            # Save model state dict
            torch.save(trainer.model.state_dict(), filepath)

            # Log model artifact if using wandb
            if isinstance(trainer.logger, WandbLogger) and trainer.logger.log_model:
                trainer.logger.experiment.save(filepath)

            if self.verbose:
                print(f"Saved last checkpoint: {filepath}")
