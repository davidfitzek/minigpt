from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class GPTConfig(BaseModel):
    """Configuration for the GPT model architecture."""

    vocab_size: int = Field(50257, description="Size of the vocabulary")
    context_length: int = Field(256, description="Maximum context length for the model")
    emb_dim: int = Field(768, description="Embedding dimension")
    n_heads: int = Field(12, description="Number of attention heads")
    n_layers: int = Field(12, description="Number of transformer layers")
    dropout: float = Field(0.0, description="Dropout rate", ge=0.0, le=1.0)
    qkv_bias: bool = Field(
        False, description="Whether to use bias in query-key-value projections"
    )

    @field_validator("n_heads")
    def validate_heads(cls, v, info):
        """Validate that embedding dimension is divisible by number of heads."""
        if "emb_dim" in info.data and info.data["emb_dim"] % v != 0:
            raise ValueError(
                f"Embedding dimension ({info.data['emb_dim']}) must be divisible by number of heads ({v})"
            )
        return v

    class Config:
        frozen = True


class TrainingSettings(BaseModel):
    """Configuration for training settings."""

    # Core training parameters
    learning_rate: float = Field(
        5e-4, description="Learning rate for the optimizer", gt=0.0
    )
    num_epochs: int = Field(10, description="Number of training epochs", gt=0)
    batch_size: int = Field(16, description="Batch size for training", gt=0)
    weight_decay: float = Field(
        0.1, description="Weight decay for the optimizer", ge=0.0
    )

    # Evaluation settings
    eval_freq: int = Field(5, description="Frequency of evaluation steps", gt=0)
    eval_iter: int = Field(1, description="Number of evaluation iterations", gt=0)
    overfit_batches: int = Field(
        0,
        description="Number of batches to overfit on for testing (0 = disabled)",
        ge=0,
    )

    # Data settings
    data_dir: str = Field(
        "src/minigpt/data/tinyshakespeare",
        description="Directory containing the dataset",
    )

    start_context: str = Field(
        "Every effort moves you", description="Starting context for text generation"
    )
    generate_samples: bool = Field(
        False, description="Whether to generate samples when evaluating"
    )

    # Reproducibility
    seed: int = Field(1, description="Random seed for reproducibility")

    # Weights & Biases settings
    wandb_project: str = Field("minigpt-training", description="W&B project name")
    wandb_run_name: Optional[str] = Field(None, description="W&B run name")
    wandb_log_freq: int = Field(100, description="W&B logging frequency", gt=0)
    wandb_enabled: bool = Field(True, description="Whether to enable W&B logging")

    @field_validator("data_dir")
    def validate_data_dir(cls, v):
        """Validate that the data directory exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Data directory does not exist: {v}")
        return v

    class Config:
        frozen = True


class MinigptConfig(BaseModel):
    """Complete configuration combining model architecture and training settings."""

    model: GPTConfig = Field(default_factory=GPTConfig)
    training: TrainingSettings = Field(default_factory=TrainingSettings)

    class Config:
        frozen = True


# Pre-defined configurations
GPT_CONFIG_124M = GPTConfig(
    vocab_size=50257,
    context_length=256,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    dropout=0.1,
    qkv_bias=False,
)

DEFAULT_TRAINING_SETTINGS = TrainingSettings(
    learning_rate=5e-4,
    num_epochs=10,
    batch_size=16,
    weight_decay=0.1,
    wandb_project="minigpt-training",
    wandb_run_name="minigpt-124M",
    seed=1,
    eval_freq=50,
    eval_iter=1,
    overfit_batches=0,  # Disabled by default
    start_context="With this work",
    generate_samples=True,
    # data_dir="src/minigpt/data/the_verdict",
    wandb_log_freq=100,
)

DEFAULT_CONFIG = MinigptConfig(
    model=GPT_CONFIG_124M,
    training=DEFAULT_TRAINING_SETTINGS,
)
