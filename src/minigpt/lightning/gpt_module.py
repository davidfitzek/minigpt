import torch
from typing import Optional, Tuple, Any

from minigpt.utils.generation_utils import generate_sample
from minigpt.lightning.lightning_module import LightningModule


class GPTLightningModule(LightningModule):
    """
    A PyTorch Lightning-style module for GPT language model training.
    This module encapsulates the model and defines the training/validation logic.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        start_context: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.start_context = start_context
        self.tokens_seen = 0

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Forward pass through the model"""
        return self.model(input_ids, targets)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Execute a single training step

        Args:
            batch: A tuple of (input_ids, target_ids)
            batch_idx: Index of the current batch

        Returns:
            loss: The training loss
        """
        del batch_idx
        inputs, targets = batch
        _, loss = self(inputs, targets)

        # Update tokens seen counter
        self.tokens_seen += inputs.numel()

        # Log metrics
        perplexity = torch.exp(loss)
        self.log("train_loss_step", loss, on_step=True)
        self.log("perplexity", perplexity, on_step=True)
        self.log("tokens_seen", self.tokens_seen, on_step=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Execute a single validation step

        Args:
            batch: A tuple of (input_ids, target_ids)
            batch_idx: Index of the current batch

        Returns:
            loss: The validation loss
        """
        del batch_idx
        inputs, targets = batch
        _, loss = self(inputs, targets)

        # Log validation metrics
        self.log("val_loss_step", loss, on_step=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure and return the optimizer

        Returns:
            optimizer: The configured optimizer
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def generate(self, device: torch.device) -> Optional[str]:
        """
        Generate text samples from the model

        Args:
            device: The device to run generation on

        Returns:
            generated_text: The generated text or None if no start context
        """
        if not self.start_context:
            return None

        return generate_sample(self.model, self.tokenizer, device, self.start_context)
