import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractDecoder(nn.Module, ABC):
    """Abstract base class for language models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Tensor of token IDs

        Returns:
            logits tensor
        """
        pass

    @abstractmethod
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, **generation_kwargs
    ) -> torch.Tensor:
        """Generate new tokens based on input tokens.

        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional generation parameters

        Returns:
            Tensor of generated token IDs including input_ids
        """
        pass
