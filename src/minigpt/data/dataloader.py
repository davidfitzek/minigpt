import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Any
import tiktoken
from torch.utils.data import DataLoader
import os
import numpy as np
from minigpt.utils.utils import read_text_file


def get_dataloader(
    version: int,
    data_dir: str,
    batch_size: int,
    max_length: int,
    train_ratio: float = 0.9,
    stride: int = 1,
    drop_last: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
):
    if version == 1:
        file_path = os.path.join(f"{data_dir}/input.txt")
        text_data = read_text_file(file_path=file_path)
        split_idx = int(train_ratio * len(text_data))
        train_loader = create_dataloader_v1(
            text_data[:split_idx],
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        val_loader = create_dataloader_v1(
            text_data[split_idx:],
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
    elif version == 2:
        file_path = os.path.join(f"{data_dir}/train")
        train_loader = create_dataloader_v2(
            file_path=file_path,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle,
        )
        file_path = os.path.join(f"{data_dir}/val")
        val_loader = create_dataloader_v2(
            file_path=file_path,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle,
        )
        file_path = os.path.join(f"{data_dir}/val")
    else:
        print("Unknown dataloader")

    return train_loader, val_loader


# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/previous_chapters.py#L157
class GPTDataset(Dataset):
    def __init__(self, txt: str, tokenizer: Any, max_length: int, stride: int) -> None:
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        token_ids: List[int] = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 1,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )
    return dataloader


# https://github.com/karpathy/nanoGPT/blob/master/train.py
class BinaryDataset(Dataset):
    """
    Dataset that efficiently loads data from binary files using memory mapping.
    This is more memory efficient than loading the entire dataset into memory.
    """

    def __init__(self, file_path: str, block_size: int, device: str = "cpu") -> None:
        self.file_path = file_path
        self.block_size = block_size
        self.device = device

        # Load the data file using memory mapping
        data_file = os.path.join(f"{file_path}.bin")
        self.data = np.memmap(data_file, dtype=np.uint16, mode="r")
        # Calculate the number of possible sequences
        self.n_sequences = len(self.data) - block_size
        print(f"Loaded dataset with {self.n_sequences} sequences")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the input sequence
        x = torch.from_numpy((self.data[idx : idx + self.block_size]).astype(np.int64))

        # Get the target sequence (shifted by 1)
        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + self.block_size]).astype(np.int64)
        )

        # Move to device if needed
        if self.device == "cuda":
            # Pin memory for faster transfer to GPU
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


def create_dataloader_v2(
    file_path: str,
    batch_size: int = 8,
    max_length: int = 256,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
) -> DataLoader:
    # Create the dataset
    dataset = BinaryDataset(
        file_path=file_path,
        block_size=max_length,
        device=device,
    )

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Set to > 0 for multi-process data loading
        drop_last=drop_last,
        pin_memory=True,  # Pin memory for faster transfer to GPU
    )

    return dataloader
