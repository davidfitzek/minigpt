#!/usr/bin/env python
import torch
import tiktoken
import argparse
from minigpt.model.transformer import GPTModel
from minigpt.config.config import DEFAULT_CONFIG


def sample_from_model(
    checkpoint_path,
    start_text="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    device=None,
):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration (using default config)
    config = DEFAULT_CONFIG.model

    # Initialize model and load weights
    model = GPTModel(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Encode the starting text
    input_ids = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0).to(device)

    # Generate text
    print(f"\nGenerating with prompt: '{start_text}'")
    print(f"Device: {device}, Temperature: {temperature}, Top-k: {top_k}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode and print the result
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print("\nGenerated Text:")
    print("=" * 40)
    print(generated_text)
    print("=" * 40)

    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample text from a trained GPT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_epoch_1.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Starting text for generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Top-k sampling (0 to disable)"
    )

    args = parser.parse_args()

    sample_from_model(
        checkpoint_path=args.checkpoint,
        start_text=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
