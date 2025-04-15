import torch
import tiktoken
import argparse
import json

from minigpt.data.dataloader import get_dataloader
from minigpt.model.transformer import GPTModel
from minigpt.config.config import MinigptConfig, DEFAULT_CONFIG
from minigpt.trainer import Trainer
from minigpt.lightning.gpt_module import GPTLightningModule
from minigpt.loggers import WandbLogger
from minigpt.callbacks import ModelCheckpoint


def main(config: MinigptConfig):
    # Setup
    torch.manual_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Initialize model components
    ##############################
    # Create the base model
    model = GPTModel(config.model)

    # Create the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the Lightning module that wraps the model
    lightning_module = GPTLightningModule(
        model=model,
        tokenizer=tokenizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        start_context=config.training.start_context,
    )

    ##############################
    # Set up dataloaders
    ##############################
    train_loader, val_loader = get_dataloader(
        version=2,
        data_dir=config.training.data_dir,
        batch_size=config.training.batch_size,
        max_length=config.model.context_length,
    )

    ##############################
    # Logger and Callback
    ##############################
    logger = WandbLogger(
        project=config.training.wandb_project,
        name=config.training.wandb_run_name,
    )

    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        # filename="minigpt_model_{epoch}",
        save_freq="step",
        every_n_steps=1,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=True,
    )

    ##############################
    # Train model
    ##############################
    trainer = Trainer(
        lightning_module=lightning_module,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MiniGPT model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (JSON)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="Number of batches to overfit on for testing (0 = disabled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
            config = MinigptConfig(**config_dict)
    else:
        config = DEFAULT_CONFIG

    # Override wandb setting if specified
    if args.no_wandb:
        config.training.wandb_enabled = False

    # Override overfit_batches if specified
    if args.overfit_batches > 0:
        config.training.overfit_batches = args.overfit_batches
        print(f"Overfitting mode enabled with {args.overfit_batches} batches")

    # Run training
    main(config)
