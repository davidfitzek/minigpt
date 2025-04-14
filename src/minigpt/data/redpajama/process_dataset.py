import os
from typing import Dict, Optional, Tuple

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Import utility functions
from minigpt.data.utils import process_text_dataset


def process_sample(
    text: str, total_tokens: int, max_tokens: Optional[int], tokenizer
) -> Tuple[str, int, bool]:
    """
    Process a single text sample.

    Args:
        text: Text to process
        total_tokens: Current token count
        max_tokens: Maximum tokens to collect
        tokenizer: Tokenizer to use

    Returns:
        Tuple of (processed_text, token_count, continue_processing)
    """
    # Skip empty or very short samples
    if not text or len(text) < 100:
        return "", 0, True

    # Encode to count tokens
    tokens = tokenizer.encode_ordinary(text)
    token_count = len(tokens)

    # Check if adding this text would exceed max_tokens
    if max_tokens is not None and total_tokens + token_count > max_tokens:
        # Only add a portion of the text to reach exactly max_tokens
        remaining_tokens = max_tokens - total_tokens
        if remaining_tokens > 0:
            partial_tokens = tokens[:remaining_tokens]
            partial_text = tokenizer.decode(partial_tokens)
            print(
                f"\nReached max_tokens limit of {max_tokens:,}. Truncated last sample."
            )
            return partial_text, remaining_tokens, False
        return "", 0, False

    # Add the full text
    return text, token_count, True


def save_combined_text(
    combined_text: str, output_dir: str, encoding: str = "gpt2"
) -> Dict:
    """
    Save the combined text to train and validation .bin files.

    Args:
        combined_text: Text to process
        output_dir: Directory to save files
        encoding: Encoding to use

    Returns:
        Dict with stats about the saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the combined text first
    input_file_path = os.path.join(output_dir, "input.txt")
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    # Use utility function to process the data into bin files
    return process_text_dataset(
        input_file=input_file_path,
        output_dir=output_dir,
        train_split=0.9,
        encoding=encoding,
    )


def process_redpajama_sample(
    output_dir: str = "redpajama_sample",
    max_tokens: Optional[int] = None,
    encoding: str = "gpt2",
) -> Dict:
    """
    Process the RedPajama-Data-1T-Sample dataset.

    Args:
        output_dir: Directory to save the processed data
        max_tokens: Maximum number of tokens to process
        encoding: Encoding to use for tokenization

    Returns:
        Dict with processing statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(encoding)

    # Load the dataset
    print("Loading RedPajama-Data-1T-Sample...")
    try:
        ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

    # Explore dataset structure
    print("\nDataset structure:")
    print(f"Features: {ds['train'].features}")
    print(f"Number of samples: {len(ds['train'])}")

    # Sample a few examples
    print("\nSample data:")
    for i, example in enumerate(ds["train"].select(range(5))):
        print(f"\nSample {i + 1}:")
        for key, value in example.items():
            if isinstance(value, str):
                print(f"{key}: {value[:100]}...")  # Show first 100 chars
            else:
                print(f"{key}: {value}")

    # Process the entire dataset
    print("\nProcessing dataset...")
    all_text = []
    total_tokens = 0

    # Process all samples
    for i, sample in enumerate(tqdm(ds["train"])):
        # Check if we've reached the max_tokens limit
        if max_tokens is not None and total_tokens >= max_tokens:
            print(f"\nReached max_tokens limit of {max_tokens:,}. Stopping processing.")
            break

        # Get the text field
        text = sample.get("text", "")

        # Process the sample
        processed_text, token_count, continue_processing = process_sample(
            text, total_tokens, max_tokens, tokenizer
        )

        if processed_text:
            all_text.append(processed_text)
            total_tokens += token_count

        if not continue_processing:
            break

        # Print progress occasionally
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples, current token count: {total_tokens:,}")

    # Combine all text
    combined_text = "\n\n".join(all_text)
    print(f"Final token count: {total_tokens:,}")

    # Save to bin files
    save_stats = save_combined_text(combined_text, output_dir, encoding)

    # Print sample text
    print("\nSample of processed text (first 500 chars):")
    print(combined_text[:500])

    return {
        "total_tokens": total_tokens,
        "train_tokens": save_stats["train_tokens"],
        "val_tokens": save_stats["val_tokens"],
        "output_dir": output_dir,
    }


def main():
    """Main entry point."""
    print("Processing RedPajama sample dataset...")

    result = process_redpajama_sample(
        output_dir="src/minigpt/data/redpajama/",
        max_tokens=100_000_000,  # Set to 1M tokens for testing
        # max_tokens=1_000_000_000,  # Uncomment for 1B tokens
    )

    if result:
        print(f"\nDataset ready at {result['output_dir']}")
        print(f"Total tokens: {result['total_tokens']:,}")
        print(f"Train tokens: {result['train_tokens']:,}")
        print(f"Val tokens: {result['val_tokens']:,}")


if __name__ == "__main__":
    main()
