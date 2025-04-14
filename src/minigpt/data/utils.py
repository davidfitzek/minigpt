# data_utils.py
import os
import requests
import tiktoken
import numpy as np


def download_file(url, filepath):
    """Download a file from a URL"""
    print(f"Downloading {url} to {filepath}")
    response = requests.get(url)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath


def process_text_dataset(input_file, output_dir, train_split=0.9, encoding="gpt2"):
    """Process a text file into train and validation binary files"""
    # Read the data
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    # Split the data
    n = len(data)
    train_data = data[: int(n * train_split)]
    val_data = data[int(n * train_split) :]

    # Encode with tokenizer
    enc = tiktoken.get_encoding(encoding)
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    # Save to binary files
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)

    return {"train_tokens": len(train_ids), "val_tokens": len(val_ids)}


def prepare_dataset(data_url, output_dir):
    """Download and prepare a text dataset"""
    # Setup paths
    input_file_path = os.path.join(output_dir, "input.txt")

    # Download if needed
    if not os.path.exists(input_file_path):
        download_file(data_url, input_file_path)
    else:
        print(f"Using existing file: {input_file_path}")

    # Process the data
    return process_text_dataset(input_file_path, output_dir)
