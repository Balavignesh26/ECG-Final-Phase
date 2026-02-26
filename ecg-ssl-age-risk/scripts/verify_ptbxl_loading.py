import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.ptbxl_dataset import PTBXLDataset
from src.utils.config import config
import pandas as pd

def verify_dataset():
    print(f"Checking data root: {config.RAW_DATA_ROOT}")
    
    csv_path = config.get_ptbxl_csv_path()
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return

    print("Loading dataframe...")
    df = pd.read_csv(csv_path, index_col='ecg_id')
    print("Columns found:", df.columns.tolist())
    
    # Check for noise columns
    noise_cols = ['static_noise', 'burst_noise', 'baseline_drift']
    present_cols = [c for c in noise_cols if c in df.columns]
    print(f"Noise columns found: {present_cols}")
    
    if present_cols:
        print("Noise stats:")
        print(df[present_cols].describe())

    print("\nInitializing Dataset (train split)...")
    try:
        ds = PTBXLDataset(split='train')
        print(f"Train size: {len(ds)}")
        
        if len(ds) > 0:
            x, y = ds[0]
            print(f"Sample 0 shape: {x.shape}")
            print(f"Sample 0 label: {y}")
            print("Successfully loaded a sample.")
        else:
            print("Dataset is empty after filtering!")
            
    except Exception as e:
        print(f"Failed to initialize/load dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset()
