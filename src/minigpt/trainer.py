import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from minigpt.utils.generation_utils import generate_sample
from minigpt.loggers import WandbLogger


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config,
        tokenizer,
        logger=None,
        callbacks=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        self.callbacks = callbacks or []  # Initialize callbacks list

        # Initialize tracking variables
        self.tokens_seen = 0
        self.global_step = -1

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
                    self.model, log="all", log_freq=config.training.wandb_log_freq
                )

    # Main training loop
    def fit(self):
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
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            if self._should_skip_batch(batch_idx):
                continue

            self._call_callbacks("on_batch_start", batch_idx=batch_idx)
            batch_loss = self._training_step(inputs, targets)
            self._call_callbacks("on_batch_end", batch_idx=batch_idx, loss=batch_loss)

            # Evaluation
            if self._should_validate():
                self._validation_epoch(epoch)

    def _training_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        _, loss = self.model(inputs, targets)

        # Optimization
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update tracking
        self.tokens_seen += inputs.numel()
        self.global_step += 1

        # Log metrics
        self._log_training_metrics(loss)

        return loss.item()

    def _validation_epoch(self, epoch: int):
        self.model.eval()
        train_loss = self._validation_loop(self.train_loader)
        val_loss = self._validation_loop(self.val_loader)
        self.model.train()

        # Log and track
        self._log_validation_metrics(train_loss, val_loss, epoch)

        # Optional text generation
        if self.config.training.generate_samples:
            self._generate_samples()

        # Notify callbacks
        self._call_callbacks(
            "on_validation_end", train_loss=train_loss, val_loss=val_loss
        )

    def _validation_loop(self, dataloader: DataLoader):
        losses = []
        num_batches = min(self.config.training.eval_iter, len(dataloader))

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break
                _, loss = self.model(inputs, targets)
                losses.append(loss.item())

        return sum(losses) / len(losses) if losses else float("nan")

    # Helper methods
    def _should_skip_batch(self, batch_idx: int):
        return (
            self.config.training.overfit_batches > 0
            and batch_idx >= self.config.training.overfit_batches
        )

    def _should_validate(self):
        return self.global_step % self.config.training.eval_freq == 0

    def _call_callbacks(self, hook_name, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                getattr(callback, hook_name)(self, **kwargs)

    # Logging methods
    def _log_training_metrics(self, loss):
        if not self.logger:
            return

        # Perplexity is a measurement of how well a language model predicts a sample of text.
        # Lower perplexity means better predictions:
        # - A perplexity of 1 means the model perfectly predicts each token
        # - A perplexity of the vocabulary size means the model is essentially random guessing
        # - For natural language, good models achieve perplexity in the range of 10-60 depending on task
        # Perplexity can be interpreted as the weighted average number of choices the model is "confused" by when predicting the next token.
        perplexity = torch.exp(loss)
        self.logger.log_metrics(
            {
                "train_loss_step": loss.item(),
                "perplexity": perplexity.item(),
                "tokens_seen": self.tokens_seen,
                "global_step": self.global_step,
            },
            step=self.global_step,
        )

    def _log_validation_metrics(self, train_loss: float, val_loss: float, epoch: int):
        if not self.logger:
            return

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
        }

        self.logger.log_metrics(metrics, step=self.global_step)

    def _generate_samples(self):
        if (
            not hasattr(self.config.training, "start_context")
            or not self.config.training.start_context
        ):
            return

        generated_text = generate_sample(
            self.model, self.tokenizer, self.device, self.config.training.start_context
        )

        if self.logger and isinstance(self.logger, WandbLogger):
            from wandb import Html

            self.logger.log_metrics(
                {
                    "generated_text": Html(
                        f"<p><b>Context:</b> {self.config.training.start_context}</p>"
                        f"<p><b>Generated:</b> {generated_text}</p>"
                    )
                },
                step=self.global_step,
            )
