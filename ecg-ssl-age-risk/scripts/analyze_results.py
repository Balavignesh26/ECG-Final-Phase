import os
import glob
import pandas as pd
import torch
from pathlib import Path

# Config
CHECKPOINT_DIR = Path(__file__).parent.parent / "experiments" / "checkpoints"

def parse_checkpoints():
    data = []
    
    # Scan for checkpoints
    # Expected structure: experiments/checkpoints/baseline/*, experiments/checkpoints/multitask/*
    
    # Baseline
    baseline_ckpts = glob.glob(str(CHECKPOINT_DIR / "baseline" / "*.ckpt"))
    for ckpt in baseline_ckpts:
        fname = os.path.basename(ckpt)
        # baseline-epoch=00-val_auroc=0.5000.ckpt
        try:
            parts = fname.split('-')
            epoch = parts[1].split('=')[1]
            val_auroc = float(parts[2].replace('.ckpt', '').split('=')[1])
            data.append({'Model': 'Baseline', 'Epoch': epoch, 'AUROC': val_auroc, 'MAE': None, 'Path': fname})
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    # Multi-Task
    multitask_ckpts = glob.glob(str(CHECKPOINT_DIR / "multitask" / "*.ckpt"))
    for ckpt in multitask_ckpts:
        fname = os.path.basename(ckpt)
        # multitask-epoch=00-val_loss_age=322.0000.ckpt
        # Note: My filename format in train_multitask only captures val_loss_age. 
        # I should probably update it to capture AUROC too or rely on loading the checkpoint to read metrics?
        # Loading is safer but slower.
        # For now, let's just list what we have.
        try:
            parts = fname.split('-')
            epoch = parts[1].split('=')[1]
            # This is specific to how I named it in train_multitask.py
            val_metric = parts[2].replace('.ckpt', '') # val_loss_age=322.0000
            val_value = float(val_metric.split('=')[1])
            
            data.append({'Model': 'MultiTask', 'Epoch': epoch, 'AUROC': None, 'MAE': val_value, 'Path': fname})
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    df = pd.DataFrame(data)
    if not df.empty:
        print(df.sort_values(by=['Model', 'Epoch']))
        df.to_csv("ablation_results.csv", index=False)
        print("Saved to ablation_results.csv")
    else:
        print("No results found.")

if __name__ == "__main__":
    parse_checkpoints()
