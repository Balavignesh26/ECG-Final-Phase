import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import PTBXLDataset
from src.utils.config import config

def plot_ecg(signal, label, lead_names=None, save_path=None):
    """
    Plots a 12-lead ECG signal.
    Args:
        signal (torch.Tensor or np.array): Shape (12, Time)
        label (Any): Diagnostic label or age
        save_path (Path or str): Path to save the plot
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
        
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axes = plt.subplots(6, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f"ECG Sample - Label/Age: {label}", fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(signal[i])
        ax.set_title(lead_names[i])
        ax.grid(True)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    print("Loading dataset...")
    # Using 100Hz for quicker visualization
    dataset = PTBXLDataset(split='train', sampling_rate=100)
    
    idx = random.randint(0, len(dataset) - 1)
    print(f"Plotting sample index: {idx}")
    
    signal, age = dataset[idx]
    
    output_dir = PROJECT_ROOT / "experiments" / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"sample_idx_{idx}_age_{age:.1f}.png"
    
    plot_ecg(signal, age, save_path=save_path)

if __name__ == "__main__":
    main()
