import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models.resnet1d import resnet1d18
from src.models.supervised_module import ECGSupervisedModule
from src.utils.config import config

def verify_model():
    print("Initializing ResNet1D-18...")
    model = resnet1d18(num_leads=12, num_classes=5)
    
    # Check parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")
    
    # Create dummy input: (Batch, Leads, Length)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 12, 2500)
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    print("Running forward pass...")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 5), f"Expected ({batch_size}, 5), got {output.shape}"
    print("ResNet1D forward pass verify: PASS")
    
    print("\nInitializing ECGSupervisedModule...")
    module = ECGSupervisedModule()
    
    # Test training step
    print("Testing training_step...")
    # Create dummy batch
    x = torch.randn(batch_size, 12, 2500)
    y = torch.randint(0, 2, (batch_size, 5)).float() # Multi-hot targets
    
    loss = module.training_step((x, y), 0)
    print(f"Training step loss: {loss.item()}")
    
    assert not torch.isnan(loss), "Loss is NaN"
    print("LightningModule step verify: PASS")

if __name__ == "__main__":
    verify_model()
