import argparse
import subprocess
import sys
from pathlib import Path

# Config
PYTHON = sys.executable
SCRIPTS_DIR = Path(__file__).parent

def run_command(cmd):
    # cmd is now a list
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def run_experiment(exp_id, data_fraction, dry_run=False, gpu_id=0):
    print(f"=== Starting Experiment {exp_id} (Data Fraction: {data_fraction}) ===")
    
    # Common Args
    # Use integers (2 batches) for limits to guarantee execution regardless of dataset size
    # max_epochs=1 is enough for dry run
    common_args = f"--limit_train_batches {1.0 if not dry_run else 2} --limit_val_batches {1.0 if not dry_run else 2} --max_epochs {50 if not dry_run else 1}".split()
    
    if exp_id == 'e1':
        # E1: Baseline Supervised (Random Init)
        cmd = [PYTHON, str(SCRIPTS_DIR / 'train_baseline.py')] + common_args + ['--data_fraction', str(data_fraction)]
        run_command(cmd)
        
    elif exp_id == 'e2':
        # E2: SSL Pre-train -> Supervised Finetune
        cmd_ssl = [PYTHON, str(SCRIPTS_DIR / 'train_ssl.py')] + common_args + ['--masking_strategy', 'random']
        run_command(cmd_ssl)
        
        print("Note: Automated chaining for E2/E4 requires checkpoint path resolution. Running dry commands only.")
        
    elif exp_id == 'e3':
        # E3: Multi-Task (Random Init)
        cmd = [PYTHON, str(SCRIPTS_DIR / 'train_multitask.py')] + common_args + ['--data_fraction', str(data_fraction)]
        run_command(cmd)

    elif exp_id == 'e4':
        # E4: SSL -> Multi-Task
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, choices=['e1', 'e2', 'e3', 'e4', 'all'], required=True)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--dry_run", action='store_true')
    args = parser.parse_args()
    
    if args.experiment == 'all':
        for e in ['e1', 'e3']: # E2/E4 need SSL integration
            run_experiment(e, args.data_fraction, args.dry_run)
    else:
        run_experiment(args.experiment, args.data_fraction, args.dry_run)
