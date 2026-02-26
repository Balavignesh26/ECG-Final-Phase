# Automated Training Pipeline Instructions

## Current Status
- ✅ **Step 1:** SSL Pre-training (50 epochs) - COMPLETED
- 🔄 **Step 2:** Baseline Training (50 epochs) - RUNNING NOW
- ⏳ **Step 3:** Multi-Task Training (30 epochs) - Will auto-start
- ⏳ **Step 4:** Ablation Study - Will auto-start

## What's Happening Automatically

The training pipeline will continue running automatically in this sequence:

### 1. Current: Baseline Training (Running)
- **Estimated completion:** ~2-3 hours from start (02:21 IST)
- **Expected finish:** ~04:30-05:30 IST

### 2. Next: Multi-Task Training (Auto-starts)
- Will load best SSL checkpoint automatically
- **Duration:** ~2 hours
- **Expected finish:** ~06:30-07:30 IST

### 3. Then: Ablation Study (Auto-starts)
Will run 6 experiments sequentially:
- E1 (Baseline) with 100%, 25%, 10% data
- E3 (Multi-Task) with 100%, 25%, 10% data
- **Duration:** ~6-8 hours total
- **Expected finish:** ~12:30-15:30 IST

### 4. Finally: Results Analysis (Auto-runs)
- Generates `ablation_results.csv`
- Compiles all metrics

## Total Estimated Completion Time
**~10-13 hours from now** (around 12:00-15:00 IST)

## What You Need to Do
**NOTHING!** Just leave your laptop:
- ✅ Plugged into power
- ✅ On a hard surface with good ventilation
- ✅ Screen can sleep (training continues in background)

## Monitoring (Optional)
If you want to check progress:
```bash
# Check GPU temperature
nvidia-smi

# Check training logs
# Look in: experiments/checkpoints/baseline/
# Look in: experiments/checkpoints/multitask/
```

## When You Return
Check `ablation_results.csv` for final comparison of all models!

---
**Note:** All training is GPU-optimized with mixed precision (fp16) for your RTX 3050.
