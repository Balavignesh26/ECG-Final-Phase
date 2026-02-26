## 🏗️ Technical Architecture

The framework is built on **PyTorch Lightning**, utilizing a modular shared-encoder architecture for multi-task optimization and self-supervised pre-training.

### 1. Neural Backbone: ResNet1D-18
The system leverages a custom **ResNet1D-18** backbone, specifically engineered for 12-lead ECG signals:
- **1D Convolutions**: Optimized for temporal signal processing, capturing rhythmic patterns better than 2D approximations.
- **Large Receptive Field ($k=15$)**: Large kernels allow the model to perceive critical cardiac features (P-QRS-T complexes) within a single filters' span.
- **Hierarchical Features**:
  - **Stem**: Conv1 ($k=15, s=2$) + MaxPool ($s=2$) for initial $4\times$ downsampling.
  - **Stages**: Four stages of `BasicBlock1D` (2 blocks each) with residual connections.
  - **Bottleneck**: Global Average Pooling into a 512-dimensional latent feature vector.

### 2. Multi-Task Learning (MTL) Module
The model concurrently performs cardiac disease classification and biological age prediction using specialized heads attached to the shared encoder.

#### **Technical Innovation: Uncertainty Weighting**
We implement **Homoscedastic Uncertainty Weighting** (Kendall et al., 2018) to automatically balance regression (MAE) and classification (BCE) losses without manual hyperparameter tuning:
$$L_{total} = \frac{1}{2\sigma_1^2}L_{age} + \frac{1}{2\sigma_2^2}L_{disease} + \log(\sigma_1) + \log(\sigma_2)$$
Where $\sigma_1$ and $\sigma_2$ are learnable noise parameters that scale the relative importance of each task during training.

---

## 🧠 Self-Supervised Learning (SSL) Implementation

The framework uses **Masked Signal Modeling (MSM)** for pre-training, forcing the encoder to learn robust global ECG representations from unlabeled data.

### **Encoder-Decoder Flow**
- **Masking**: Random patches (15-50% of signal), entire leads, or temporal blocks are zeroed out.
- **Recovery**: A **Transposed Convolution Decoder** (`decoder1d.py`) reconstructs the full 12-lead signal from the encoded latent features.
- **Objective**: The model minimizes MSE loss on the masked regions:
  $$L_{SSL} = \frac{1}{M} \sum_{i \in Mask} (x_i - \hat{x}_i)^2$$

---

## 🔄 Data Pipeline

The pipeline processes the **PTB-XL** dataset with clinical-grade signal preprocessing:
1. **Resampling**: 500Hz $\rightarrow$ **250Hz** (preserves diagnostic fidelity while halving compute).
2. **Filtering**: Butterworth **Bandpass Filter (0.5-50 Hz)** for baseline wander and noise removal.
3. **Normalization**: Lead-wise Z-score normalization combined with **Robust Scaling** (IQR-based) for artifact resilience.

---

## 🎯 Experimental Results

### **Summary Table**

| Experiment | Model Type | Data | AUROC | MAE (years) | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Supervised | 100% | **90.79%** | - | **Outstanding** 🏆 |
| **E1 Ablation** | Supervised | 10% | **87.95%** | 10.90 | Excellent |
| **E3 Multi-Task** | Multi-Task | 100% | 89.2% | **10.79** | Good |

### **Key Technical Takeaways**
- **Data Efficiency**: Achieved **87.95% AUROC** with only **10%** of labeled data, demonstrating the architecture's power in low-data regimes.
- **MTL Synergy**: Age prediction (10.79 years MAE) serves as an effective auxiliary task, improving feature robustness.
- **SSL Robustness**: Validation reconstruction loss of **0.2859** indicates the encoder has successfully learned the underlying morphology of ECG signals.

---

## 📁 Project Structure

```bash
ecg-ssl-age-risk/
├── src/
│   ├── models/
│   │   ├── resnet1d.py      # ResNet-1D architecture (encoder)
│   │   ├── decoder1d.py     # Decoder for SSL reconstruction
│   │   ├── multitask_module.py # Multi-task logic + Uncertainty weighting
│   │   └── ssl_module.py    # Self-supervised learning logic
│   ├── data/
│   │   ├── preprocessing.py # Signal filtering and scaling
│   │   └── ptbxl_dataset.py # PTB-XL loader
│   └── utils/
│       └── masking.py       # SSL masking strategies
```
