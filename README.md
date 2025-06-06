# SSL-MAE: Adaptive Semi-Supervised Learning Framework for Multi-Label Classification of Remote Sensing Images Using Masked Autoencoders

This repository contains the official implementation of our paper **"SSL-MAE: Adaptive Semi-Supervised Learning Framework for Multi-Label Classification of Remote Sensing Images Using Masked Autoencoders"**.

**Authors:** Marjan Stoimchev, Jurica Levatić, Dragi Kocev, Sašo Džeroski

## Abstract

The increasing volume of remotely sensed imagery (RSI) requires efficient processing and extraction of meaningful information. Modern deep learning architectures excel in various tasks but typically require large labeled datasets, which are often scarce in RSI due to the tedious labeling of complex heterogeneous landscapes containing multiple semantic categories. This can limit the potential of supervised deep learning methods. To address this, we propose SSL-MAE, a novel semi-supervised learning method based on a masked autoencoder. Our approach unifies self-supervision and discriminative learning within a single, end-to-end framework, leveraging both abundant unlabeled data and limited labeled data. Additionally, we introduce an adaptive mechanism to control the level of supervision during learning, crucial for balancing prediction quality with effective use of unlabeled data.

## Methodology

<img id="methodology-overview" style="height:400px;width:800px;" src="images/methodology_v3.pdf" alt="SSL-MAE Framework Overview" />

SSL-MAE integrates self-supervised and supervised learning within a unified masked autoencoder framework consisting of five key components:

### 1. Image Masking
Each input image (labeled or unlabeled) is divided into non-overlapping patches, with a fraction randomly masked to create a reconstruction-based pretext task that drives robust representation learning from partial visual data.

### 2. Vision Transformer Encoder
A DeiT-based encoder processes visible patches into latent embeddings, leveraging global modeling capabilities of self-attention for large-scale visual data processing.

### 3. Classification Head
Only latent representations from unmasked patches of labeled images are passed to a lightweight classification head. These embeddings are aggregated using average pooling and processed through a fully connected layer with sigmoid activation for multi-label classification.

### 4. Lightweight Decoder
Following SimMIM design, a lightweight decoder reconstructs pixel intensities of masked patches using ℓ₁ regression loss for both labeled and unlabeled samples, providing direct self-supervised signal for enhanced feature learning.

### 5. Adaptive Joint Learning
Model parameters are updated via weighted combination of supervised and unsupervised losses: