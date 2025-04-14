from minigpt.model.transformer import GPTModel
from minigpt.config.config import GPT_CONFIG_124M
import torch


def test_model():
    config = GPT_CONFIG_124M
    model = GPTModel(config)
    print("compiling...")
    model = torch.compile(model)
    print("done")
    run_gpt_model_tests(model)


def run_gpt_model_tests(model, model_name="GPT model"):
    """Common test function for both uncompiled and compiled GPT models"""
    # Test input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))

    # Test forward pass without labels
    logits, _ = model(input_ids)

    # Check output shapes
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == 1  # Only the last position is returned
    assert logits.shape[2] == model.config.vocab_size

    # Test forward pass with labels
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
    logits, loss = model(input_ids, labels)

    # Check output shapes with labels
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == seq_length
    assert logits.shape[2] == model.config.vocab_size

    # Check that loss is calculated
    assert loss is not None
    assert loss.item() > 0

    # Test with different sequence lengths
    short_seq_length = 5
    short_input_ids = torch.randint(
        0, model.config.vocab_size, (batch_size, short_seq_length)
    )
    with torch.no_grad():
        logits, _ = model(short_input_ids)

    # Check output shapes for shorter sequence
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == 1
    assert logits.shape[2] == model.config.vocab_size

    # Test the generate method
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 5))
    max_new_tokens = 10

    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_tokens)

    # Check that the generated sequence has the expected length
    assert generated_ids.shape[0] == batch_size
    assert generated_ids.shape[1] == 5 + max_new_tokens

    print(f"All tests passed for {model_name}!")


if __name__ == "__main__":
    test_model()
