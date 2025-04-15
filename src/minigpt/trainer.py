import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Optional, Any

from minigpt.loggers import WandbLogger
from minigpt.callbacks import Callback


class Trainer:
    def __init__(
        self,
        lightning_module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Any,
        logger: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.lightning_module = lightning_module.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = lightning_module.configure_optimizers()
        self.device = device
        self.config = config
        self.logger = logger
        self.callbacks = callbacks or []

        # Initialize tracking variables
        self.tokens_seen = 0
        self.global_step = -1
        self.val_losses = []
        self.train_losses = []

        # For tracking running training loss
        self.current_epoch_losses = []

        # Log hyperparameters if logger exists
        if self.logger:
            # Log model config and training config
            self.logger.log_hyperparams(
                {
                    "model": self.config.model.model_dump(),
                    "training": self.config.training.model_dump(),
                }
            )

            # Watch model gradients
            if hasattr(self.logger, "watch"):
                self.logger.watch(
                    self.lightning_module.model,
                    log="all",
                    log_freq=config.training.wandb_log_freq,
                )

    # Main training loop
    def fit(self):
        """Main training method that orchestrates the entire training process"""
        # Setup phase
        self._call_callbacks("on_fit_start")

        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self._call_callbacks("on_epoch_start", epoch=epoch)
            self._train_epoch(epoch)
            self._call_callbacks("on_epoch_end", epoch=epoch)

        # Cleanup phase
        self._call_callbacks("on_fit_end")
        if self.logger:
            self.logger.finalize()

    # Training epoch
    def _train_epoch(self, epoch: int):
        """Handle training for one epoch"""
        self.lightning_module.train()
        self.current_epoch_losses = []  # Reset epoch losses

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            if self._should_skip_batch(batch_idx):
                continue

            self._call_callbacks("on_batch_start", batch_idx=batch_idx)
            batch_loss = self._training_step(batch)
            self._call_callbacks("on_batch_end", batch_idx=batch_idx, loss=batch_loss)

            # Validation
            if self._should_validate():
                self._validation_epoch(epoch)

        # At the end of the epoch, calculate and store average loss
        if self.current_epoch_losses:
            avg_train_loss = sum(self.current_epoch_losses) / len(
                self.current_epoch_losses
            )
            self.train_losses.append(avg_train_loss)

    def _training_step(self, batch: torch.Tensor):
        """Process a single training batch"""
        # Get the loss from the lightning module, passing the batch index
        batch_idx = self.global_step + 1  # Next step index
        loss = self.lightning_module.training_step(batch, batch_idx)

        # Optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update global step
        self.global_step += 1

        # Track batch loss for epoch average
        loss_value = loss.item()
        self.current_epoch_losses.append(loss_value)

        # Process and log metrics
        self._process_logged_metrics()

        return loss_value

    def _validation_epoch(self, epoch: int):
        """Run validation and log metrics"""
        self.lightning_module.eval()

        # Only validate on the validation dataset
        val_loss = self._validation_loop(self.val_loader)
        self.val_losses.append(val_loss)

        # Get the average training loss from the current epoch
        avg_train_loss = 0.0
        if self.current_epoch_losses:
            avg_train_loss = sum(self.current_epoch_losses) / len(
                self.current_epoch_losses
            )

        # Process any metrics logged during validation
        self._process_logged_metrics()

        # Set model back to training mode
        self.lightning_module.train()

        # Optional text generation
        if self.config.training.generate_samples:
            self._generate_samples()

        # Notify callbacks
        self._call_callbacks(
            "on_validation_end", train_loss=avg_train_loss, val_loss=val_loss
        )

    def _validation_loop(self, dataloader: DataLoader):
        """Run validation on the given dataloader"""
        losses = []
        num_batches = min(self.config.training.eval_iter, len(dataloader))

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                # Run validation step
                loss = self.lightning_module.validation_step(batch, i)
                losses.append(loss.item())
                self._process_logged_metrics()

        # Notify the lightning module that validation is complete
        if hasattr(self.lightning_module, "on_validation_epoch_end"):
            self.lightning_module.on_validation_epoch_end()
            # Process any metrics logged during the end hook
            self._process_logged_metrics()

        return sum(losses) / len(losses) if losses else float("nan")

    # Helper methods
    def _should_skip_batch(self, batch_idx: int):
        """Check if batch should be skipped (for overfit mode)"""
        return (
            self.config.training.overfit_batches > 0
            and batch_idx >= self.config.training.overfit_batches
        )

    def _should_validate(self):
        """Check if validation should be performed"""
        return self.global_step % self.config.training.eval_freq == 0

    def _call_callbacks(self, hook_name, **kwargs):
        """Call the specified hook on all callbacks"""
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                getattr(callback, hook_name)(self, **kwargs)

    def _process_logged_metrics(self):
        """Process and log metrics from the lightning module"""
        # Early return if no logger or no metrics
        if not self.logger:
            return

        metrics = self.lightning_module.get_logged_metrics()
        if not metrics:
            return

        # Start with common metrics
        metrics_to_log = {"global_step": self.global_step}

        # Extract values for metrics that should be logged
        metrics_to_log.update(
            {name: info["value"] for name, info in metrics.items() if info["logger"]}
        )

        # Log metrics and reset
        self.logger.log_metrics(metrics_to_log, step=self.global_step)
        self.lightning_module.reset_logged_metrics()

    def _generate_samples(self):
        """Generate and log text samples from the model"""
        generated_text = self.lightning_module.generate(self.device)

        if generated_text and self.logger and isinstance(self.logger, WandbLogger):
            from wandb import Html

            context = self.config.training.start_context
            self.logger.log_metrics(
                {
                    "generated_text": Html(
                        f"<p><b>Context:</b> {context}</p>"
                        f"<p><b>Generated:</b> {generated_text}</p>"
                    )
                },
                step=self.global_step,
            )
