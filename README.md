# Human Segmentation with PyTorch and EfficientNet

![Title Image](title-image.png)

### Overview

This project implements an end-to-end human image segmentation pipeline using PyTorch. It trains a model to accurately separate human figures from backgrounds in images, enabling applications like photo editing, virtual try-ons, and AR filters. The code covers data loading, augmentation, training, evaluation, and inference in a Jupyter notebook.

## Live Demo
Try the deployed app here:  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nauman123-coder-human-segmentation-pytorch-efficientnet-4563d0.streamlit.app)

### What is the Model?

The model is a binary segmentation network that predicts pixel-wise masks for humans (white) vs. background (black). It uses a U-Net architecture, which excels at capturing both local details and global context for precise segmentation.

### Why Do We Need It?

Human segmentation is essential for real-world tasks where isolating people from scenes is required, such as in social media apps for background replacement or in security systems for person detection. This efficient model provides high accuracy with low computational cost, making it suitable for deployment on edge devices.

## Data

* **Dataset**: Human-Segmentation-Dataset (loaded from CSV with image and mask paths).
* **Size**: 290 images split into train (232) and validation (58) sets.
* **Preprocessing**: Images resized to 320x320, normalized to [0,1]; masks are binary (grayscale, expanded to 3D for processing).
* **Augmentations**: Training uses resize, horizontal/vertical flips (via Albumentations); validation uses resize only.

## Model Architecture

* **Backbone**: U-Net from segmentation_models_pytorch library.
* **Encoder**: timm-efficientnet-b0, pretrained on ImageNet (weights='imagenet').
* **Input**: RGB images (3 channels, 320x320).
* **Output**: Single-channel logits for binary masks.
* **Loss**: Combined Dice Loss + Binary Cross-Entropy (BCEWithLogitsLoss).
* **Optimizer**: Adam with learning rate 0.003.
* **Training**: 25 epochs, batch size 16, on CUDA (GPU).
* **Inference**: Sigmoid activation + threshold (0.5) for binary masks.

