import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.models.ssl_module import ECGSSLModule
from src.data.ptbxl_dataset import PTBXLDataset
from src.utils.masking import mask_random_patches

def visualize(args):
    # Load Model
    if args.ckpt_path:
        print(f"Loading model from {args.ckpt_path}")
        model = ECGSSLModule.load_from_checkpoint(args.ckpt_path)
    else:
        print("Initializing untrained model for demonstration...")
        model = ECGSSLModule(masking_strategy='random')
    
    model.eval()
    
    # Load Data
    ds = PTBXLDataset(split='val')
    x, _ = ds[0] # (12, 2500)
    x = x.unsqueeze(0) # (1, 12, 2500)
    
    # Forward
    with torch.no_grad():
        # Mask
        masked_x, mask = mask_random_patches(x, mask_ratio=0.5)
        
        # Recon
        # We need to manually call internal methods because forward() returns features
        start_enc = model.encoder.forward_features(masked_x)
        recon_x = model.decoder(start_enc)
        
    # Plot
    # Plot Lead I (Index 0)
    lead_idx = 0
    t = np.arange(x.shape[2])
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Original
    axes[0].plot(t, x[0, lead_idx].numpy(), label='Original', color='black')
    axes[0].set_title("Original Signal (Lead I)")
    axes[0].legend()
    
    # Masked
    # Highlight masked regions
    masked_sig = masked_x[0, lead_idx].numpy()
    mask_sig = mask[0, lead_idx].numpy()
    
    axes[1].plot(t, masked_sig, label='Masked Input', color='blue')
    # Overlay removed parts in red? or just show gaps
    # Let's show the mask in background
    axes[1].fill_between(t, -2, 2, where=mask_sig>0, color='red', alpha=0.1, label='Masked Regions')
    axes[1].set_title("Masked Input")
    axes[1].legend()
    
    # Reconstructed
    axes[2].plot(t, x[0, lead_idx].numpy(), label='Original', color='black', alpha=0.3)
    axes[2].plot(t, recon_x[0, lead_idx].numpy(), label='Reconstructed', color='green')
    axes[2].set_title("Reconstruction")
    axes[2].legend()
    
    plt.tight_layout()
    output_path = "reconstruction_sample.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    
    visualize(args)
