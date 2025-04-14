import json
import tempfile

from minigpt.config.config import (
    GPTConfig,
    TrainingSettings,
    MinigptConfig,
    DEFAULT_CONFIG,
)


def test_gpt_config_basic():
    """Basic sanity check for GPTConfig."""
    # Test default initialization
    config = GPTConfig()
    assert config.vocab_size == 50257
    assert config.n_heads == 12

    # Test custom values
    custom_config = GPTConfig(vocab_size=1000, n_heads=8, emb_dim=512)
    assert custom_config.vocab_size == 1000
    assert custom_config.n_heads == 8


def test_gpt_config_validation():
    """Test validation in GPTConfig."""
    # Should fail - embedding dimension not divisible by heads
    try:
        GPTConfig(emb_dim=100, n_heads=3)
        assert False, "Should have failed with validation error"
    except Exception as e:
        assert "divisible by number of heads" in str(e)


def test_training_settings():
    """Basic sanity check for TrainingSettings."""
    # Create a temporary directory to use as data_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test initialization with custom values
        settings = TrainingSettings(learning_rate=0.001, num_epochs=5, data_dir=tmpdir)

        assert settings.learning_rate == 0.001
        assert settings.num_epochs == 5
        assert settings.data_dir == tmpdir

        # Test validation fails with non-existent directory
        try:
            TrainingSettings(data_dir="/path/does/not/exist")
            assert False, "Should have failed with validation error"
        except Exception as e:
            assert "directory does not exist" in str(e)


def test_minigpt_config():
    """Test the combined MinigptConfig."""
    # Create a temporary directory to use as data_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model and training configs
        model_config = GPTConfig(vocab_size=1000, emb_dim=256, n_heads=4)
        training_config = TrainingSettings(
            learning_rate=0.001, num_epochs=5, data_dir=tmpdir
        )

        # Combine into MinigptConfig
        config = MinigptConfig(model=model_config, training=training_config)

        # Verify attributes are correctly accessible
        assert config.model.vocab_size == 1000
        assert config.model.emb_dim == 256
        assert config.model.n_heads == 4
        assert config.training.learning_rate == 0.001
        assert config.training.num_epochs == 5

        # Test JSON serialization/deserialization
        config_json = config.json()
        config_dict = json.loads(config_json)
        restored_config = MinigptConfig(**config_dict)

        assert restored_config.model.vocab_size == 1000
        assert restored_config.training.learning_rate == 0.001



def test_default_config():
    """Test the predefined DEFAULT_CONFIG."""
    assert isinstance(DEFAULT_CONFIG, MinigptConfig)
    assert DEFAULT_CONFIG.model.vocab_size == 50257
    assert DEFAULT_CONFIG.model.n_layers == 12



if __name__ == "__main__":
    print("Running simple config tests...")

    try:
        test_gpt_config_basic()
        test_gpt_config_validation()
        test_training_settings()
        test_minigpt_config()
        test_default_config()

        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
