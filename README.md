# NYCU Visual Recognition using Deep Learning - Spring 2026 - Homework 1

## Introduction

This repository contains a high-performance image classification pipeline built in PyTorch, designed to classify images into 100 distinct categories. To maximize representational efficiency within a strict 100-million parameter limit, the core architecture utilizes a custom **ResNet-101** backbone enhanced with **Squeeze-and-Excitation (SE)** channel attention modules.

The training and inference pipelines are heavily optimized to prevent overfitting and ensure robust generalization on unseen test data, particularly for datasets with limited samples (e.g., ~20,000 images). 

Key optimizations include:
* **Progressive Transfer Learning:** The model utilizes pre-trained ImageNet weights with a targeted layer unfreezing schedule. Combined with a `ReduceLROnPlateau` scheduler, this allows the network to adapt to the new dataset without catastrophically forgetting its foundational edge and texture detectors.
* **Balanced Regularization:** Implementation of Label Smoothing and light MixUp data augmentation to explicitly penalize overconfident predictions and smooth class boundaries, without destroying critical spatial features.
* **Stochastic Weight Averaging (SWA):** Averaging model weights over the final training epochs to ensure the network settles in a flat, robust minimum, significantly reducing leaderboard variance.
* **Robust Inference:** A multiprocessing-safe 10-Crop Test-Time Augmentation (TTA) strategy combined with Soft-Voting Ensembling to mathematically fuse the predictive confidence of multiple checkpoints.

## Environment Setup

### Dependencies

This project requires **Python 3.11** and is optimized for GPU acceleration using **PyTorch with CUDA 12.4**.

1. **Create and activate a virtual environment (recommended):**
```bash
conda create -n resnet_env python=3.11
conda activate resnet_env
```
   
2. Install PyTorch and TorchVision with CUDA 12.4 support:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. Install additional dependencies:   
```bash
pip install matplotlib tqdm pillow tensorboard
```

## Usage

### 1. Dataset Structure

Ensure your dataset is organized in standard PyTorch `ImageFolder` format within the `./data` directory:

Plaintext

```
data/
├── train/
│   ├── 0/
│   ├── ...
│   └── 99/
├── valid/
│   ├── 0/
│   ├── ...
│   └── 99/
└── test/
    ├── test_image_1.jpg
    └── ...
```

### 2. Training the Model

To initiate the training pipeline, execute the `train.py` script. This will automatically handle dataset loading, apply the OneCycle learning rate schedule, and transition into the SWA phase during the final 25% of epochs.

```
python train.py
```

- **Outputs:** The script generates a `class_mapping.pth` file for inference decoding, logs metrics to TensorBoard (`runs/`), and saves two checkpoints: the single best epoch (`best_custom_resnet101_model.pth`) and the averaged weights (`best_swa_resnet101_model.pth`).
- **Monitoring:** The script tracks metrics using TensorBoard. You can view real-time training and validation curves by running `tensorboard --logdir=runs` in a separate terminal.

### 3. Running Inference

To generate predictions on the test set, you can use either the standard inference script or the ensemble script for maximum accuracy.

**Standard 10-Crop Inference:** Targets a single model checkpoint (defaults to the SWA model) and applies 10-Crop TTA.

```bash
python inference.py
```

**Ensemble Inference (Recommended):** Loads both the best single-epoch baseline model and the SWA model into memory, running 10-Crop TTA on both and fusing their logits via Soft Voting.

```bash
python ensemble_inference.py
```

- **Outputs:** Both scripts will parse the test directory and generate a `prediction.csv` file mapping each image filename to its predicted class string.

## Performance Snapshot
![Screenshot 2026-03-30 223224_censored](https://github.com/user-attachments/assets/d2da8ec8-4224-4fe2-baaf-b113b4c5ba24)

