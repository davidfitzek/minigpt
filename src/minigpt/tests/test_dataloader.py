import os
import torch
import tiktoken
from minigpt.data.dataloader import (
    GPTDataset,
    BinaryDataset,
    create_dataloader_v1,
    create_dataloader_v2,
)


def test_gpt_dataset():
    """Simple test for GPTDataset"""
    print("\nTesting GPTDataset...")

    # Test text and parameters
    test_text = "Hello world! This is a simple test for the GPT dataset."
    max_length = 8
    stride = 1
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDataset(test_text, tokenizer, max_length, stride)
    print(f"Dataset length: {len(dataset)}")

    # Assert dataset is not empty
    assert len(dataset) > 0, "Dataset should not be empty"

    # Check first item
    input_ids, target_ids = dataset[0]
    print(f"Input shape: {input_ids.shape}, Target shape: {target_ids.shape}")

    # Assert shapes are correct
    assert input_ids.shape == (max_length,), f"Input shape should be ({max_length},)"
    assert target_ids.shape == (max_length,), f"Target shape should be ({max_length},)"

    # Verify target is shifted input
    assert torch.all(input_ids[1:] == target_ids[:-1]), "Target should be shifted input"
    print("Target verification passed: target is shifted input")

    # Check a few random items
    for i in range(min(3, len(dataset))):
        idx = torch.randint(0, len(dataset), (1,)).item()
        input_ids, target_ids = dataset[idx]
        assert input_ids.shape == (max_length,), (
            f"Input shape should be ({max_length},) for item {idx}"
        )
        assert target_ids.shape == (max_length,), (
            f"Target shape should be ({max_length},) for item {idx}"
        )
        assert torch.all(input_ids[1:] == target_ids[:-1]), (
            f"Target should be shifted input for item {idx}"
        )

    # Decode for visual inspection
    print(f"Input: {tokenizer.decode(input_ids.tolist())}")
    print(f"Target: {tokenizer.decode(target_ids.tolist())}")

    print("GPTDataset test completed")


def test_binary_dataset():
    """Simple test for BinaryDataset"""
    print("\nTesting BinaryDataset...")

    # Check if Shakespeare dataset exists
    data_dir = "src/minigpt/data/tinyshakespeare"
    train_file = os.path.join(data_dir, "train.bin")

    if not os.path.exists(train_file):
        print(f"Warning: Shakespeare dataset not found at {train_file}")
        print("Skipping BinaryDataset test")
        return

    # Create dataset
    block_size = 128
    dataset = BinaryDataset(
        file_path=os.path.join(data_dir, "train"), block_size=block_size, device="cpu"
    )

    print(f"Dataset length: {len(dataset)}")

    # Assert dataset is not empty
    assert len(dataset) > 0, "Dataset should not be empty"

    # Check first item
    input_ids, target_ids = dataset[0]
    print(f"Input shape: {input_ids.shape}, Target shape: {target_ids.shape}")

    # Assert shapes are correct
    assert input_ids.shape == (block_size,), f"Input shape should be ({block_size},)"
    assert target_ids.shape == (block_size,), f"Target shape should be ({block_size},)"

    # Verify target is shifted input
    assert torch.all(input_ids[1:] == target_ids[:-1]), "Target should be shifted input"
    print("Target verification passed: target is shifted input")

    # Check a few random items
    for i in range(min(3, len(dataset))):
        idx = torch.randint(0, len(dataset), (1,)).item()
        input_ids, target_ids = dataset[idx]
        assert input_ids.shape == (block_size,), (
            f"Input shape should be ({block_size},) for item {idx}"
        )
        assert target_ids.shape == (block_size,), (
            f"Target shape should be ({block_size},) for item {idx}"
        )
        assert torch.all(input_ids[1:] == target_ids[:-1]), (
            f"Target should be shifted input for item {idx}"
        )

    # Decode for visual inspection
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Input: {tokenizer.decode(input_ids.tolist())}")
    print(f"Target: {tokenizer.decode(target_ids.tolist())}")

    print("BinaryDataset test completed")


def test_dataloaders():
    """Simple test for dataloader creation functions"""
    print("\nTesting dataloader creation functions...")

    # Test text for v1 dataloader
    test_text = "Hello world! This is a simple test for the dataloader."

    # Test v1 dataloader
    batch_size = 2
    max_length = 8
    dataloader_v1 = create_dataloader_v1(
        txt=test_text, batch_size=batch_size, max_length=max_length, stride=1
    )
    print(f"v1 Dataloader created with {len(dataloader_v1)} batches")

    # Assert dataloader is not empty
    assert len(dataloader_v1) > 0, "v1 Dataloader should not be empty"

    # Check first batch
    for inputs, targets in dataloader_v1:
        print(f"Batch shapes - Input: {inputs.shape}, Target: {targets.shape}")

        # Assert batch shapes are correct
        assert inputs.shape == (batch_size, max_length), (
            f"Input batch shape should be ({batch_size}, {max_length})"
        )
        assert targets.shape == (batch_size, max_length), (
            f"Target batch shape should be ({batch_size}, {max_length})"
        )

        # Assert targets are shifted inputs
        for i in range(batch_size):
            assert torch.all(inputs[i, 1:] == targets[i, :-1]), (
                f"Target should be shifted input for batch item {i}"
            )

        print(f"First item in batch - Input: {inputs[0]}, Target: {targets[0]}")
        break

    # Test v2 dataloader if Shakespeare dataset exists
    data_dir = "src/minigpt/data/tinyshakespeare"
    train_file = os.path.join(data_dir, "train.bin")

    if os.path.exists(train_file):
        dataloader_v2 = create_dataloader_v2(
            file_path=os.path.join(data_dir, "train"),
            batch_size=batch_size,
            max_length=max_length,
        )
        print(f"v2 Dataloader created with {len(dataloader_v2)} batches")

        # Assert dataloader is not empty
        assert len(dataloader_v2) > 0, "v2 Dataloader should not be empty"

        # Check first batch
        for inputs, targets in dataloader_v2:
            print(f"Batch shapes - Input: {inputs.shape}, Target: {targets.shape}")

            # Assert batch shapes are correct
            assert inputs.shape == (batch_size, max_length), (
                f"Input batch shape should be ({batch_size}, {max_length})"
            )
            assert targets.shape == (batch_size, max_length), (
                f"Target batch shape should be ({batch_size}, {max_length})"
            )

            # Assert targets are shifted inputs
            for i in range(batch_size):
                assert torch.all(inputs[i, 1:] == targets[i, :-1]), (
                    f"Target should be shifted input for batch item {i}"
                )

            print(f"First item in batch - Input: {inputs[0]}, Target: {targets[0]}")
            break
    else:
        print("Shakespeare dataset not found, skipping v2 dataloader test")

    print("Dataloader tests completed")


if __name__ == "__main__":
    test_gpt_dataset()
    test_binary_dataset()
    test_dataloaders()
