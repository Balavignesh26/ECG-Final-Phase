import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models.ssl_module import ECGSSLModule
from src.utils.masking import mask_random_patches, mask_leads, mask_temporal_block

def verify_ssl():
    print("Initializing ECGSSLModule...")
    module = ECGSSLModule(masking_strategy='random')
    
    # Check parameter count
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,}")
    
    # Create dummy input: (Batch, Leads, Length)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 12, 2500)
    print(f"Input shape: {input_tensor.shape}")
    
    # Test Masking
    print("Testing Masking...")
    masked_x, mask = mask_random_patches(input_tensor)
    print(f"Masked shape: {masked_x.shape}, Mask shape: {mask.shape}")
    assert masked_x.shape == input_tensor.shape
    assert mask.shape == input_tensor.shape
    
    # Test Forward (Encoder features)
    print("Testing Forward (Feature Extraction)...")
    feats = module(input_tensor)
    print(f"Features shape: {feats.shape}")
    # ResNet18 spatial features should be (B, 512, ~79)
    
    # Test Training Step (Full Recon Loop)
    print("Testing Training Step...")
    loss = module.training_step((input_tensor, None), 0)
    print(f"Training step loss: {loss.item()}")
    
    # Test Decoder Output Shape via internal call
    recon = module.decoder(feats)
    print(f"Reconstruction shape: {recon.shape}")
    assert recon.shape == input_tensor.shape, f"Recon shape mismatch: {recon.shape} vs {input_tensor.shape}"
    
    print("SSL Module verify: PASS")

if __name__ == "__main__":
    verify_ssl()
