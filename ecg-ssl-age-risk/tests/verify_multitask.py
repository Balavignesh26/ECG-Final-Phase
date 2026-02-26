import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models.multitask_module import ECGMultiTaskModule

def verify_multitask():
    print("Initializing ECGMultiTaskModule...")
    module = ECGMultiTaskModule()
    
    # Check Heads
    print("Checking Heads...")
    print(f"Age Head: {module.age_head}")
    print(f"Disease Head: {module.disease_head}")
    print(f"Log Vars (Uncertainty Params): {module.log_vars.data}")
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 12, 2500)
    
    # Create dummy targets
    # Dataset now returns dict
    age_y = torch.randn(batch_size) # Ages
    disease_y = torch.randint(0, 2, (batch_size, 5)).float() # Multi-label
    
    target = {
        'age': age_y,
        'disease': disease_y
    }
    
    # Test Forward
    print("Testing Forward...")
    age_pred, disease_logits = module(x)
    print(f"Age Pred Shape: {age_pred.shape}")
    print(f"Disease Logits Shape: {disease_logits.shape}")
    
    assert age_pred.shape == (batch_size, 1)
    assert disease_logits.shape == (batch_size, 5)
    
    # Test Training Step
    print("Testing Training Step...")
    loss = module.training_step((x, target), 0)
    print(f"Training Loss: {loss.item()}")
    
    # Check if gradients flow to log_vars
    loss.backward()
    print(f"Log Vars Grads: {module.log_vars.grad}")
    assert module.log_vars.grad is not None, "Log vars not learning!"
    
    print("Multi-Task Module verify: PASS")

if __name__ == "__main__":
    verify_multitask()
