# MiniGPT: A Minimal LLM Training Framework

MiniGPT is a lightweight training framework designed for large language models (LLMs). It provides a clean, modular implementation of a GPT-style transformer model with all essential components needed for training and evaluation.

## Features

- **Modular Design**: Clear abstractions and interfaces for easy extension
- **Full Training Pipeline**: Includes dataloader, trainer, loss functions, and metrics logging
- **Lightning-style Components**: Implementation inspired by PyTorch Lightning patterns
- **Flexible Configuration**: Pydantic-based configuration with validation
- **Monitoring**: Integrated with Weights & Biases for experiment tracking
- **Checkpointing**: Automatic saving of model states during training

## Installation

### Create Virtual Environment

```bash
uv venv
```

### Install Dependencies

```bash
source .venv/bin/activate && uv pip install -e .
```

## Usage

### Running Tests

Verify the codebase is working correctly:

```bash
pytest src/minigpt/tests
```

### Training a Model

Run a training experiment:

```bash
python examples/training.py --config src/minigpt/config/gpt_config.json
```

Additional options:
- `--no-wandb`: Disable Weights & Biases logging
- `--overfit-batches N`: Overfit on N batches (useful for debugging)

### Customizing Training

You can modify the configuration file (`gpt_config.json`) to change model parameters, training settings, and more.

## Framework Architecture

The framework follows a modular design pattern with clear separation of concerns:

![architecture](assets/architecture.png)

Key components:
- **Dataloader**: Handles data loading and preprocessing
- **Model**: Implements the transformer architecture
- **LightningModule**: Encapsulates training/validation logic
- **Trainer**: Orchestrates the training loop
- **Callbacks**: Provides hooks for custom behavior
- **Loggers**: Tracks metrics and experiment progress

## Model Architecture

MiniGPT implements a GPT-2 style transformer architecture:

![miniGPT](assets/gpt2.webp)

[Source](https://medium.com/@vipul.koti333/from-theory-to-code-step-by-step-implementation-and-code-breakdown-of-gpt-2-model-7bde8d5cecda)

The model includes:
- Token and positional embeddings
- Multi-head self-attention layers
- Feed-forward networks
- Layer normalization
- Dropout regularization

## Example Training Runs

You can view an example training run on the RedPajama dataset:

Run 1: [Wandb: Redpajama dataset, 100M samples, 124M params transformer, 1000 train steps](https://wandb.ai/quantcomp/minigpt-training/runs/rzgt7510/workspace?nw=nwuserquantdavidos)

Run 2: [Wandb: Redpajama dataset, 100M samples, 124M params transformer, 8000 train steps](https://wandb.ai/quantcomp/minigpt-training/runs/8m9c7xjp?nw=nwuserquantdavidos)

Run 3: [Wandb: Redpajama dataset, 100M samples, 124M params transformer, 8000 train steps](https://wandb.ai/quantcomp/minigpt-training/runs/xxmwyxuc?nw=nwuserquantdavidos)

## Project Structure

```
src/
└── minigpt/
    ├── callbacks/          # Training callbacks (checkpointing, etc.)
    ├── config/             # Configuration definitions
    ├── data/               # Data handling and datasets
    ├── lightning/          # Lightning-style modules
    ├── loggers/            # Logging implementations
    ├── model/              # Model architecture
    ├── tests/              # Unit tests
    ├── trainer.py          # Main trainer implementation
    └── utils/              # Utility functions
```

## Relevant Resources

This project draws inspiration from:
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/03_bonus_pretraining_on_gutenberg/previous_chapters.py) 
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the framework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.