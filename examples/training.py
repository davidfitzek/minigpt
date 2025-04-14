import torch
import tiktoken
import argparse
import json

from minigpt.data.dataloader import get_dataloader
from minigpt.model.transformer import GPTModel
from minigpt.config.config import MinigptConfig, DEFAULT_CONFIG
from minigpt.trainer import Trainer
from minigpt.loggers import WandbLogger
from minigpt.callbacks import ModelCheckpoint


def main(
    config: MinigptConfig):
    # Setup
    torch.manual_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Initialize model, tokenizer and optimizer
    ##############################

    model = GPTModel(config.model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    tokenizer = tiktoken.get_encoding("gpt2")

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
        dirpath="checkpoints",  # Where to save the checkpoints
        filename="minigpt_model_{epoch}",  # Checkpoint filename pattern
        save_freq="epoch",  # Save at the end of each epoch
        every_n_epochs=1,  # Save every epoch
        save_best_only=True,  # Only save when the model improves
        monitor="val_loss",  
        mode="min",  # Lower is better for loss
        verbose=True,  # Print info when saving
    )

    ##############################
    # Train model
    ##############################

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        tokenizer=tokenizer,
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
