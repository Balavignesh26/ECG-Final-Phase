import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.datamodule import ECGDataModule
from src.utils.config import config
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def verify_pipeline():
    print("Initializing DataModule...")
    dm = ECGDataModule(batch_size=32, num_workers=0)
    dm.setup()
    
    # 1. Check Split Sizes
    train_len = len(dm.train_dataset)
    val_len = len(dm.val_dataset)
    test_len = len(dm.test_dataset)
    
    print(f"Train size: {train_len}")
    print(f"Val size: {val_len}")
    print(f"Test size: {test_len}")
    
    assert train_len > 0, "Train set is empty"
    assert val_len > 0, "Val set is empty"
    assert test_len > 0, "Test set is empty"
    
    # 2. Check Overlap using indices
    train_indices = set(dm.train_dataset.df.index)
    val_indices = set(dm.val_dataset.df.index)
    test_indices = set(dm.test_dataset.df.index)
    
    assert train_indices.isdisjoint(val_indices), "Train and Val sets overlap!"
    assert train_indices.isdisjoint(test_indices), "Train and Test sets overlap!"
    assert val_indices.isdisjoint(test_indices), "Val and Test sets overlap!"
    print("Split overlap check passed.")
    
    # 3. Check Batch Shapes and Types
    print("Checking Train DataLoader...")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    x, y = batch
    
    print(f"Batch X shape: {x.shape}") # Should be (32, 12, 2500)
    print(f"Batch Y shape: {y.shape}") # Should be (32, 5)
    
    expected_samples = int(config.TARGET_SAMPLING_RATE * config.DURATION)
    assert x.shape == (32, 12, expected_samples), f"Expected ({32}, {12}, {expected_samples}), got {x.shape}"
    assert y.shape == (32, 5), f"Expected ({32}, {5}), got {y.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    
    print("Shape and Type checks passed.")
    
    # 4. Check Label Distribution in Train
    print("Checking label distribution (first 1000 samples)...")
    # We can access the dataframe directly for efficiency
    label_counts = dm.train_dataset.df['label_vector'].apply(lambda x: pd.Series(x)).sum()
    label_counts.index = config.DIAGNOSTIC_SUPERCLASSES
    print(label_counts)
    
    print("\nPipeline verified successfully!")

if __name__ == "__main__":
    verify_pipeline()
