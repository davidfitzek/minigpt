import torch
from minigpt.trainer import Trainer
from minigpt.config.config import MinigptConfig, GPTConfig, TrainingSettings

# Simple dummy model for testing
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        
    def forward(self, x, targets=None):
        logits = self.linear(x)
        loss = torch.nn.functional.mse_loss(logits, targets) if targets is not None else None
        return logits, loss

# Test utilities
def assert_equal(a, b, message=None):
    assert a == b, f"Expected {a} to equal {b}. {message or ''}"

def assert_not_equal(a, b, message=None):
    assert a != b, f"Expected {a} to not equal {b}. {message or ''}"

def assert_true(condition, message=None):
    assert condition, f"Expected condition to be True. {message or ''}"

# Simple test functions
def test_initialization():
    print("Testing trainer initialization...")
    
    # Create real config (with test overrides)
    training_settings = TrainingSettings(
        num_epochs=1,
        eval_freq=2,
        eval_iter=2,
        batch_size=4,
        overfit_batches=0,
        generate_samples=True,
        wandb_enabled=False,  # Disable wandb for testing
        start_context="Test context",
        data_dir="."  
    )
    model_config = GPTConfig(
        vocab_size=100,
        context_length=20,
        emb_dim=12,
        n_heads=3,
        n_layers=2
    )
    config = MinigptConfig(model=model_config, training=training_settings)
    
    # Create test objects
    model = DummyModel()
    train_loader = [(torch.randn(2, 10), torch.randn(2, 10)) for _ in range(3)]
    val_loader = [(torch.randn(2, 10), torch.randn(2, 10)) for _ in range(2)]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cpu')
    tokenizer = None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        tokenizer=tokenizer
    )
    
    # Check initialization
    assert_equal(trainer.global_step, -1, "Initial global step should be -1")
    assert_equal(trainer.tokens_seen, 0, "Initial tokens seen should be 0")
    assert_equal(len(trainer.callbacks), 0, "Initial callbacks should be empty")
    
    print("✓ Initialization test passed")
    return trainer, train_loader, config

def test_training_step(trainer, train_loader):
    print("Testing training step...")
    
    # Get a sample batch
    inputs, targets = train_loader[0]
    
    # Store initial parameters
    initial_params = next(trainer.model.parameters()).clone().detach()
    
    # Perform training step
    loss = trainer._training_step(inputs, targets)
    
    # Check results
    current_params = next(trainer.model.parameters())
    assert_not_equal(torch.sum(initial_params).item(), torch.sum(current_params).item(), 
                    "Model parameters should change after training step")
    assert_equal(trainer.global_step, 0, "Global step should be incremented")
    assert_equal(trainer.tokens_seen, inputs.numel(), "Tokens seen should be updated")
    assert_true(isinstance(loss, float), "Loss should be a float")
    
    print("✓ Training step test passed")

def test_validation_loop(trainer):
    print("Testing validation loop...")
    
    # Create simple validation data
    val_data = [(torch.randn(2, 10), torch.randn(2, 10)) for _ in range(3)]
    
    # Run validation
    loss = trainer._validation_loop(val_data)
    
    # Check results
    assert_true(isinstance(loss, float), "Validation loss should be a float")
    assert_true(not torch.isnan(torch.tensor(loss)), "Validation loss should not be NaN")
    
    print("✓ Validation loop test passed")

def test_callback_mechanism(trainer):
    print("Testing callback mechanism...")
    
    # Create a simple callback object
    class TestCallback:
        def __init__(self):
            self.called = False
            self.args = None
            
        def test_hook(self, trainer, **kwargs):
            self.called = True
            self.args = kwargs
    
    callback = TestCallback()
    trainer.callbacks = [callback]
    
    # Call the callback
    trainer._call_callbacks("test_hook", param1="test")
    
    # Check results
    assert_true(callback.called, "Callback should have been called")
    assert_equal(callback.args, {"param1": "test"}, "Callback should receive correct arguments")
    
    print("✓ Callback mechanism test passed")


def test_model():
    print("Running trainer tests...\n")
    
    trainer, train_loader, config = test_initialization()
    test_training_step(trainer, train_loader)
    test_validation_loop(trainer)
    test_callback_mechanism(trainer)
    
    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    test_model()