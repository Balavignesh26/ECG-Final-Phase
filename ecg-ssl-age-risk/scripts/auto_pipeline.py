import subprocess
import time
import sys
from pathlib import Path
import glob

def wait_for_process(process_name):
    """Wait for a specific Python process to complete"""
    print(f"Waiting for {process_name} to complete...")
    while True:
        try:
            # Check if process is still running
            result = subprocess.run(
                ['tasklist', '/FI', f'WINDOWTITLE eq {process_name}'],
                capture_output=True,
                text=True
            )
            if process_name not in result.stdout:
                print(f"{process_name} completed!")
                break
        except:
            pass
        time.sleep(60)  # Check every minute

def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in a directory"""
    ckpt_files = glob.glob(str(checkpoint_dir / "*.ckpt"))
    if not ckpt_files:
        return None
    # Sort by modification time, get latest
    latest = max(ckpt_files, key=lambda x: Path(x).stat().st_mtime)
    return latest

def run_command(cmd, description):
    """Run a command and log output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n⚠ WARNING: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully!")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return True

def main():
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"
    
    print("="*60)
    print("AUTOMATED TRAINING PIPELINE")
    print("="*60)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will automatically run:")
    print("1. Wait for current baseline training to finish")
    print("2. Multi-Task Training (30 epochs)")
    print("3. Ablation Study (E1-E4 with 100%, 25%, 10% data)")
    print("4. Results Analysis")
    print("="*60)
    
    # Step 1: Wait for baseline to complete (already running)
    print("\n[STEP 1] Baseline training is already running...")
    print("Monitoring will happen externally. Proceeding to next steps when ready.\n")
    
    # For automation, we'll just wait a bit and then check for checkpoint
    # In practice, you'd monitor the process more carefully
    input("\nPress ENTER when baseline training completes to continue...")
    
    # Step 2: Multi-Task Training with SSL weights
    print("\n[STEP 2/3] Starting Multi-Task Training...")
    ssl_ckpt_dir = project_root / "experiments" / "checkpoints" / "ssl"
    ssl_ckpt = find_latest_checkpoint(ssl_ckpt_dir)
    
    if ssl_ckpt:
        print(f"Found SSL checkpoint: {ssl_ckpt}")
        cmd = [
            sys.executable,
            str(scripts_dir / "train_multitask.py"),
            "--max_epochs", "30",
            "--ssl_ckpt", str(ssl_ckpt)
        ]
    else:
        print("No SSL checkpoint found, training from scratch")
        cmd = [
            sys.executable,
            str(scripts_dir / "train_multitask.py"),
            "--max_epochs", "30"
        ]
    
    if not run_command(cmd, "Multi-Task Training"):
        print("Stopping pipeline due to error.")
        return
    
    # Step 3: Ablation Study
    print("\n[STEP 3/3] Starting Ablation Study...")
    
    experiments = [
        ("E1 - Baseline 100%", ["e1", "1.0"]),
        ("E1 - Baseline 25%", ["e1", "0.25"]),
        ("E1 - Baseline 10%", ["e1", "0.1"]),
        ("E3 - Multi-Task 100%", ["e3", "1.0"]),
        ("E3 - Multi-Task 25%", ["e3", "0.25"]),
        ("E3 - Multi-Task 10%", ["e3", "0.1"]),
    ]
    
    for desc, (exp_id, data_frac) in experiments:
        cmd = [
            sys.executable,
            str(scripts_dir / "run_ablation.py"),
            "--experiment", exp_id,
            "--data_fraction", data_frac
        ]
        run_command(cmd, desc)
    
    # Step 4: Analyze Results
    print("\n[STEP 4/4] Analyzing Results...")
    cmd = [sys.executable, str(scripts_dir / "analyze_results.py")]
    run_command(cmd, "Results Analysis")
    
    print("\n" + "="*60)
    print("FULL PIPELINE COMPLETED!")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nResults saved to: ablation_results.csv")
    print("Checkpoints saved in: experiments/checkpoints/")

if __name__ == "__main__":
    main()
