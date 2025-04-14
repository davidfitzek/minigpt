import torch.nn as nn
import time
from minigpt.loggers import WandbLogger


def test_wandb_logger():
    """Simple test function for the WandbLogger."""
    print("Testing WandbLogger...")

    # Create a simple model for testing
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    # Initialize the logger with a test project
    logger = WandbLogger(
        project="minigpt-test", name=f"logger-test-{int(time.time())}", log_model=False
    )

    # Test logging hyperparameters
    print("Logging hyperparameters...")
    hyperparams = {"learning_rate": 0.001, "batch_size": 32, "model_type": "test_model"}
    logger.log_hyperparams(hyperparams)

    # Test watching model
    print("Watching model...")
    logger.watch(model, log="gradients", log_freq=10)

    # Test logging metrics
    print("Logging metrics...")
    for step in range(5):
        metrics = {
            "train_loss": 1.0 / (step + 1),
            "val_loss": 1.5 / (step + 1),
            "accuracy": 0.5 + step * 0.1,
        }
        logger.log_metrics(metrics, step=step)
        print(f"Logged metrics at step {step}: {metrics}")
        time.sleep(1)  # Small delay to see the logs in wandb

    # Test finalizing
    print("Finalizing logger...")
    logger.finalize()

    print("WandbLogger test completed!")


if __name__ == "__main__":
    test_wandb_logger()
