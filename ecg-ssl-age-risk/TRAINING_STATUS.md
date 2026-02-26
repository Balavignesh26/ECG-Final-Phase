# Training Pipeline Status

**Last Updated:** 2026-02-13 21:05 IST

## ✅ Completed Experiments

### 1. SSL Pre-training (50 epochs)
- **Status:** ✅ COMPLETE
- **Best Model:** `ssl/resnet1d-ssl-epoch=49-val_loss=0.2859.ckpt`
- **Final Loss:** 0.2859
- **Assessment:** ✅ **Excellent**

### 2. Baseline Training (50 epochs)
- **Status:** ✅ COMPLETE (Early Stopped at Epoch 12)
- **Best Model:** `baseline/resnet1d-epoch=10-val_auroc=0.9079.ckpt`
- **Best AUROC:** **90.79%** 🎉
- **Assessment:** ✅ **Outstanding** (90%+ is excellent in medical ML)

### 3. E1 Ablation - 100% Data
- **Status:** ✅ COMPLETE (50 epochs)
- **AUROC:** 89.9%
- **MAE:** 10.90 years
- **Assessment:** ✅ **Excellent**

## 🔄 Currently Running

### 4. Ablation Study - Remaining Experiments
- **Status:** 🔄 Running E1 with 25% data (Experiment 1/5)
- **Started:** 21:05 IST
- **Queue:**
  - [🔄] E1 (Baseline) - 25% data (Running)
  - [⏳] E1 (Baseline) - 10% data
  - [⏳] E3 (Multi-Task) - 100% data (50 epochs)
  - [⏳] E3 (Multi-Task) - 25% data
  - [⏳] E3 (Multi-Task) - 10% data
- **Total Remaining:** 5 experiments
- **Estimated Duration:** 10-12 hours
- **Estimated Completion:** Tomorrow ~07:00-09:00 IST

## 📊 Overall Progress

```
[████████████░░░░░░░░] 60% Complete

✅ SSL Pre-training (Excellent)
✅ Baseline Training (Outstanding - 90.79% AUROC)
✅ E1 Ablation - 100% (Excellent - 89.9% AUROC)
🔄 E1 Ablation - 25% (Running)
⏳ E1 Ablation - 10% (Queued)
⏳ E3 Multi-Task - 100% (Queued)
⏳ E3 Multi-Task - 25% (Queued)
⏳ E3 Multi-Task - 10% (Queued)
```

## 🎯 Results Summary

| Experiment | Data | AUROC | MAE | Assessment |
|------------|------|-------|-----|------------|
| SSL Pre-train | 100% | - | - | ✅ Excellent (val_loss=0.2859) |
| Baseline | 100% | **90.79%** | - | ✅ **Outstanding** |
| E1 Ablation | 100% | 89.9% | 10.90 | ✅ Excellent |
| E1 Ablation | 25% | - | - | 🔄 Running |
| E1 Ablation | 10% | - | - | ⏳ Pending |
| E3 Multi-Task | 100% | - | - | ⏳ Pending |
| E3 Multi-Task | 25% | - | - | ⏳ Pending |
| E3 Multi-Task | 10% | - | - | ⏳ Pending |

## ⏰ Timeline

- **Original Start:** 2026-02-13 00:42 IST
- **Resumed:** 2026-02-13 21:05 IST
- **Current Phase:** Ablation Study (Experiment 1/5)
- **Estimated Remaining:** 10-12 hours
- **Expected Finish:** Tomorrow 07:00-09:00 IST

---

**Ablation study running automatically - leave laptop plugged in overnight!**

