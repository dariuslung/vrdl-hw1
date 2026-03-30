# NYCU Visual Recognition using Deep Learning - Spring 2026 - HW1

## Introduction

This repository contains a highly optimized, custom image classification pipeline designed to classify images across 100 distinct categories. Built on a PyTorch framework, the core model utilizes a modified **ResNet-50 backbone** integrated with **Squeeze-and-Excitation (SE) attention mechanisms** at every bottleneck stage. 

Designed to maximize representational efficiency while strictly adhering to a sub-100 million parameter budget (operating at ~26M parameters), the pipeline includes several advanced training and inference strategies to prevent overfitting on moderately sized datasets (~20k input data size):

* **Progressive Unfreezing:** Leverages ImageNet pre-trained weights, gradually unfreezing deeper residual layers while decaying the learning rate to adapt to the specific dataset without catastrophic forgetting.
* **Aggressive Regularization:** Employs `RandAugment` and `RandomErasing` during training, coupled with a heavily regularized, custom dropout classification head.
* **Class Balancing:** Utilizes a weighted random sampler to ensure balanced batch generation across all 100 classes.
* **Test-Time Augmentation (TTA):** Averages predictions from multi-view inputs (original and horizontally flipped) during inference to boost generalization and stability.

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
└── valid/
    ├── 0/
    ├── ...
    └── 99/
```

### 2. Training the Model

To initiate the training pipeline with progressive unfreezing and data augmentation, execute the training script:

```bash
python train.py
```

- **Monitoring:** The script tracks metrics using TensorBoard. You can view real-time training and validation curves by running `tensorboard --logdir=runs` in a separate terminal.
- **Outputs:** `best_custom_resnet50_model.pth`: The saved weights of the model at its lowest validation loss.
    - `class_mapping.pth`: A serialized dictionary mapping tensor indices to human-readable class names.
    - `training_metrics.png`: A static plot of the loss and accuracy over all epochs.

### 3. Running Inference

To generate predictions on the unlabelled test set using the trained weights and Test-Time Augmentation, run:
```bash
python inference.py
```

- **Outputs:** This will evaluate all images in the `./data/test` directory and generate a `prediction.csv` file containing two columns: `image_name` and `pred_label`.

## Performance Snapshot
<img width="1264" height="513" alt="Screenshot 2026-03-26 213604_censored" src="https://github.com/user-attachments/assets/f99aa46f-0c82-447c-b2bc-76b2b8fd3b54" />
