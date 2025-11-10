# SSL-MAE: Adaptive Semi-Supervised Learning Framework for Multi-Label Classification of Remote Sensing Images Using Masked Autoencoders

This repository contains the official implementation of our paper **"SSL-MAE: Adaptive Semi-Supervised Learning Framework for Multi-Label Classification of Remote Sensing Images Using Masked Autoencoders"**.

**Authors:** Marjan Stoimchev, Jurica Levatić, Dragi Kocev, Sašo Džeroski

## Abstract

The increasing volume of remotely sensed imagery (RSI) requires efficient processing and extraction of meaningful information. Modern deep learning architectures excel in various tasks but typically require large labeled datasets, which are often scarce in RSI due to the tedious labeling of complex heterogeneous landscapes containing multiple semantic categories. This can limit the potential of supervised deep learning methods. To address this, we propose SSL-MAE, a novel semi-supervised learning method based on a masked autoencoder. Our approach unifies self-supervision and discriminative learning within a single, end-to-end framework, leveraging both abundant unlabeled data and limited labeled data. Additionally, we introduce an adaptive mechanism to control the level of supervision during learning, crucial for balancing prediction quality with effective use of unlabeled data.

(The code will soon be updated)

## Methodology

<div align="center">
  <img src="media/methodology.png" alt="SSL-MAE Framework Overview" width="100%" style="max-width: 1000px; height: auto;" />
</div>

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



We investigate two strategies for the weight parameter w:
- **Grid Search (SSL-MAE-GS)**: Optimal w selected via validation performance
- **Learnable Weight (SSL-MAE-wₗ)**: w parameterized as trainable sigmoid function

The adaptive mechanism allows the network to automatically balance reliance on labeled vs unlabeled data, enabling effective exploitation of abundant unlabeled data while emphasizing discriminative learning as needed.

## Key Features

- **Unified End-to-End Framework**: Combines self-supervised and supervised learning without two-stage training
- **Adaptive Supervision Control**: Dynamic balancing of supervision levels during training
- **Multi-Label Support**: Specifically designed for complex multi-label remote sensing classification
- **Transferable Design**: Adaptive joint learning can enhance other self-supervised methods
- **Data Efficient**: Effective performance with limited labeled data (as low as 1%)

## Installation

```bash
git clone https://github.com/marjanstoimchev/SSL-MAE.git
cd SSL-MAE

# Create conda environment
conda create -n ssl-mae python=3.8
conda activate ssl-mae

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
SSL-MAE/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   └── ssl_mae.py           # SSL-MAE model architecture
│   ├── datasets/                 # Dataset loaders for remote sensing datasets
│   │   ├── dataset_selector.py
│   │   ├── ucm_dataset.py
│   │   ├── aid_dataset.py
│   │   └── ...
│   ├── utils/                    # Utility functions
│   │   ├── dataset_utils.py     # Dataset splitting and sampling
│   │   ├── simmim_utils.py      # SimMIM masking utilities
│   │   └── ...
│   ├── configs/                  # Configuration management
│   │   └── config.py            # Dataset configs and argument parser
│   └── trainers/                 # Training modules
│       ├── learner.py           # PyTorch Lightning trainer
│       ├── fabric_learner.py    # Lightning Fabric trainer
│       └── callbacks.py         # Custom callbacks
├── scripts/                      # Training and inference scripts
│   ├── train_lightning.py       # PyTorch Lightning training
│   ├── train_fabric.py          # Lightning Fabric training
│   ├── inference.py
│   └── inference_lightning.py
├── media/                        # Methodology diagrams
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset Preparation

SSL-MAE supports multiple remote sensing datasets:

### Supported Datasets

**Multi-Label Classification (MLC):**
- **UCM_mlc** - UC Merced (17 classes)
- **AID_mlc** - Aerial Image Dataset (17 classes)
- **MLRSNet** - (60 classes)
- **DFC_15** - (8 classes)
- **BEN_43** - BigEarthNet (43 classes)
- **Ankara**

**Multi-Class Classification (MCC):**
- **UCM_mcc** - UC Merced (21 classes)
- **AID_mcc** - Aerial Image Dataset (30 classes)
- **RESISC45** - (45 classes)
- **RSSCN7** - (7 classes)
- **OPTIMAL-31** - (31 classes)

### Dataset Organization

Place your datasets in the following structure:
```
../rs_datasets/
├── mlc/                    # Multi-label datasets
│   ├── UCM_mlc/
│   ├── AID_mlc/
│   └── ...
└── mcc/                    # Multi-class datasets
    ├── UCM_mcc/
    ├── AID_mcc/
    └── ...
```

## Training

### Quick Start with PyTorch Lightning (Recommended)

```bash
python scripts/train_lightning.py \
    --learning_task mlc \
    --dataset UCM_mlc \
    --epochs 100 \
    --batch_size 16 \
    --mask_ratio 0.3 \
    --architecture deit \
    --model_size base \
    --mode semi_supervised \
    --seed 42
```

### Training with Lightning Fabric

For lower-level control over training:

```bash
python scripts/train_fabric.py \
    --learning_task mlc \
    --dataset UCM_mlc \
    --epochs 100 \
    --batch_size 16 \
    --mask_ratio 0.3 \
    --architecture deit \
    --model_size base \
    --seed 42
```

### Command-Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--learning_task` | Classification task type | `mlc` | `mlc`, `mcc` |
| `--dataset` | Dataset name | `UCM_mlc` | See supported datasets |
| `--epochs` | Number of training epochs | `10` | Any integer |
| `--batch_size` | Batch size | `16` | Any integer |
| `--mask_ratio` | Masking ratio for MAE | `0.3` | 0.0-1.0 |
| `--architecture` | Backbone architecture | `deit` | `deit`, `vit` |
| `--model_size` | Model size | `base` | `tiny`, `small`, `base`, `large` |
| `--mode` | Training mode | `semi_supervised` | `supervised`, `semi_supervised` |
| `--seed` | Random seed | `42` | Any integer |

### Configuring Label Fraction

Control the percentage of labeled data in semi-supervised learning by modifying the `fraction_labeled` parameter in the training scripts:

```python
# In train_lightning.py or train_fabric.py
if __name__ == '__main__':
    args = ConfigSelector()
    main(args, fraction_labeled=0.1, w=None)  # 10% labeled data
```

### Weight Parameter (`w`)

The weight parameter `w` controls the balance between reconstruction and classification losses:

- **`w=None`** (default): Learnable weight (SSL-MAE-wₗ) - automatically learned during training
- **`w=0.5`**: Fixed equal weighting
- **`w` ∈ [0, 1]**: Custom fixed weight (0 = only reconstruction, 1 = only classification)

## Inference

Run inference on trained models:

```bash
python scripts/inference_lightning.py \
    --checkpoint path/to/checkpoint.ckpt \
    --dataset UCM_mlc \
    --learning_task mlc
```

## Advanced Usage

### Multi-GPU Training

PyTorch Lightning automatically handles multi-GPU training. Modify the `devices` parameter in `train_lightning.py`:

```python
trainer = L.Trainer(
    devices=[0, 1, 2, 3],  # Use GPUs 0, 1, 2, 3
    strategy='ddp',
    ...
)
```

### Hyperparameter Configuration

Key hyperparameters can be adjusted in `src/configs/config.py`:

```python
class BaseConfig:
    lr: float = 1e-3              # Learning rate
    min_lr: float = 1e-5           # Minimum learning rate
    warmup_epochs: int = 5         # Warmup epochs
    weight_decay: float = 1e-6     # Weight decay
    n_accumulate: int = 8          # Gradient accumulation steps
    patience: int = 10             # Early stopping patience
```

### Adding Custom Datasets

1. Create a dataset loader in `src/datasets/my_dataset.py`
2. Register it in `src/configs/config.py`:

```python
datasets = {
    "mlc": {
        "MyDataset": DatasetConfig(
            image_size=224,
            n_classes=10,
            extension="jpg"
        ),
    }
}
```

## Monitoring

Training is logged to Weights & Biases (wandb):

```bash
wandb login
# View logs at https://wandb.ai/your-username/SSL-MAE-lightning
```

Saved checkpoints and logs:
- `saved_models/` - PyTorch Lightning checkpoints
- `saved_models_fabric/` - Lightning Fabric checkpoints

## Troubleshooting

**CUDA Out of Memory:**
- Reduce `--batch_size`
- Increase `n_accumulate` for gradient accumulation
- Use smaller model: `--model_size tiny` or `--model_size small`

**Import Errors:**
Ensure scripts are run from the repository root directory.

**Dataset Not Found:**
Verify dataset path structure: `../rs_datasets/{learning_task}/{dataset}`

## Citation
```bash
@article{stoimchev2024ssl_mae,
  title={SSL-MAE: Adaptive Semi-Supervised Learning Framework for Multi-Label Classification of Remote Sensing Images Using Masked Autoencoders},
  author={Stoimchev, Marjan and Levatić, Jurica and Kocev, Dragi and Džeroski, Sašo},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025}
}
```
