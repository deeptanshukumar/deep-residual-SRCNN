# Deep Residual SRCNN — Image Super-Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of single-image super-resolution (SISR) using progressively deeper and residual SRCNN architectures. This project explores the impact of network depth, training stabilization, and residual learning on super-resolution quality.

## Project Overview

This study investigates **single-image super-resolution** using convolutional neural networks. Starting from conventional interpolation methods and the original SRCNN as baselines, we progressively explore:

- **Deep SRCNN** — Extended 5-layer architecture
- **Stabilized Deep SRCNN** — Lower learning rate + gradient clipping
- **Deep Residual SRCNN** — Global residual connection for learning reconstruction residuals
- **Data-Scale Study** — 60 vs. 800 training images

## Results

| Method | Training Images | Test MSE | Test PSNR |
|--------|----------------|----------|-----------|
| Nearest Neighbor | — | 0.003608 | 24.43 dB |
| Bilinear | — | 0.003663 | 24.36 dB |
| Bicubic | — | 0.002985 | 25.25 dB |
| SRCNN | 60 | 0.002848 | 25.46 dB |
| Deep SRCNN (stabilized) | 60 | 0.002894 | 25.38 dB |
| Deep Residual SRCNN | 60 | 0.001919 | 27.17 dB |
| Deep Residual SRCNN | 800 | 0.001674 | 27.76 dB |

## Architecture

### SRCNN (Baseline)
```
Input (3×224×224) → Conv(3→64, 9×9) → ReLU → Conv(64→32, 1×1) → ReLU → Conv(32→3, 5×5) → Output
```
**Parameters:** 20,099

### Deep SRCNN
```
Input → Conv(3→64, 9×9) → ReLU → Conv(64→64, 3×3) → ReLU → Conv(64→64, 3×3) → ReLU → Conv(64→32, 1×1) → ReLU → Conv(32→3, 5×5) → Output
```
**Parameters:** 93,955

### Deep Residual SRCNN
```
Input → Conv(3→64, 9×9) → ReLU → Conv(64→64, 3×3) → ReLU → Conv(64→64, 3×3) → ReLU → Conv(64→32, 1×1) → ReLU → Conv(32→3, 5×5) → Output
Output = Input + Residual
```
**Parameters:** 93,955

## Project Structure

```
deep-residual-srcnn/
├── code/
│   ├── config.py          # Configuration and hyperparameters
│   ├── data.py            # Data loading and preprocessing
│   ├── models.py          # Model architectures (SRCNN, DeepSRCNN, DeepResidualSRCNN)
│   ├── metrics.py         # PSNR and MSE calculations
│   ├── train.py           # Training loop with checkpointing
│   ├── evaluate.py        # Model evaluation
│   ├── baselines.py       # Interpolation baselines
│   ├── visualize.py       # Visualization utilities
│   └── main.py            # Main entry point
├── outputs/               # Generated results and plots
├── data/                  # Downloaded datasets
├── material/              # Reference papers and materials
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/deep-residual-srcnn.git
cd deep-residual-srcnn
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
torchvision
numpy
pandas
matplotlib
pillow
pyarrow
```

## Usage

### Run All Experiments

```bash
cd code
python main.py
```

This will:
1. Download the DIV2K dataset (100 images)
2. Train SRCNN, Deep SRCNN, and Deep Residual SRCNN
3. Evaluate all models and baselines
4. Generate comparison plots and save checkpoints
5. Optionally run the full 800-image dataset experiment

### Run Individual Components

```python
from data import load_100_image_dataset, create_dataloaders
from models import DeepResidualSRCNN
from train import train_model
from evaluate import evaluate_model

# Load data
df, all_hr, all_lr_small, all_lr, train_idx, val_idx, test_idx = load_100_image_dataset()
train_loader, val_loader, test_loader = create_dataloaders(all_hr, all_lr, train_idx, val_idx, test_idx)

# Train model
model = DeepResidualSRCNN().to(device)
history = train_model(model, train_loader, val_loader, device, lr=3e-4, grad_clip_norm=1.0)

# Evaluate
mse, psnr = evaluate_model(model, test_loader, device)
print(f"Test MSE: {mse:.6f}, Test PSNR: {psnr:.2f} dB")
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | MSE |
| Batch Size | 4 |
| Epochs | 50 |
| Learning Rate (SRCNN) | 1e-3 |
| Learning Rate (Deep/Residual) | 3e-4 |
| Gradient Clipping | 1.0 |
| Image Size | 224×224 |
| LR Size | 112×112 |

## Key Findings

1. **Residual learning provides the largest single improvement** — Deep Residual SRCNN achieves 27.17 dB vs. 25.46 dB for baseline SRCNN (+1.71 dB)

2. **Deeper architecture alone does not guarantee improvement** — Deep SRCNN without stabilization underperforms the baseline

3. **Training stabilization matters** — Lower learning rate + gradient clipping enables deeper models to train effectively

4. **More training data helps** — Increasing from 60 to 800 training images yields +0.59 dB improvement

## License

This project is licensed under the MIT License.

## Acknowledgments

- Original SRCNN paper: Dong et al., "Image Super-Resolution Using Deep Convolutional Networks" (2014)
- DIV2K Dataset: ETH Zurich
