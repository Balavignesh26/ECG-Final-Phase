import torch
import numpy as np

def mask_random_patches(x, mask_ratio=0.5, patch_size=50):
    """
    Randomly mask patches of the signal.
    Args:
        x: (Batch, Leads, Length)
        mask_ratio: Percentage of signal to mask
        patch_size: Size of contiguous mask in samples
    Returns:
        masked_x: tensor with masked regions zeroed out (or replaced)
        mask: boolean tensor (1 = masked, 0 = visible)
    """
    B, C, L = x.shape
    num_patches = L // patch_size
    num_mask = int(mask_ratio * num_patches)
    
    mask = torch.zeros(B, num_patches, device=x.device)
    
    # Randomly select patches to mask
    for i in range(B):
        mask_idx = torch.randperm(num_patches)[:num_mask]
        mask[i, mask_idx] = 1
        
    # Upsample mask to original length
    # Shape: (B, num_patches) -> (B, 1, num_patches) -> (B, 1, L)
    mask = mask.unsqueeze(1).repeat_interleave(patch_size, dim=2)
    
    # Handle remainder if L is not perfectly divisible by patch_size 
    if mask.shape[2] < L:
        padding = torch.zeros(B, 1, L - mask.shape[2], device=x.device)
        mask = torch.cat([mask, padding], dim=2)
    
    # Broadcast to all leads
    mask = mask.repeat(1, C, 1) # (B, C, L)
    
    masked_x = x.clone()
    masked_x[mask == 1] = 0 # Zero out masked regions
    
    return masked_x, mask

def mask_leads(x, num_leads_to_mask=1):
    """
    Mask entire leads randomly.
    Args:
        x: (Batch, Leads, Length)
        num_leads_to_mask: Number of leads to zero out
    Returns:
        masked_x: tensor
        mask: boolean tensor
    """
    B, C, L = x.shape
    mask = torch.zeros(B, C, L, device=x.device)
    
    for i in range(B):
        leads_idx = torch.randperm(C)[:num_leads_to_mask]
        mask[i, leads_idx, :] = 1
        
    masked_x = x.clone()
    masked_x[mask == 1] = 0
    
    return masked_x, mask

def mask_temporal_block(x, block_size=500):
    """
    Mask a single contiguous block of time across all leads.
    Args:
        x: (Batch, Leads, Length)
        block_size: Size of the block
    Returns:
        masked_x: tensor
        mask: boolean tensor
    """
    B, C, L = x.shape
    mask = torch.zeros(B, C, L, device=x.device)
    
    for i in range(B):
        start = np.random.randint(0, L - block_size)
        mask[i, :, start:start+block_size] = 1
        
    masked_x = x.clone()
    masked_x[mask == 1] = 0
    
    return masked_x, mask
