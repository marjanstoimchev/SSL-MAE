import math
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.dataset import SSLMAEDataset, load_hf_splits
from src.data.transforms import SimMIMTransform

logger = logging.getLogger(__name__)


# --- Splitting utilities ---

def _stratified_split_mlc(labels, test_size=0.2, seed=42):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(msss.split(np.zeros(len(labels)), labels))
    except ImportError:
        logger.warning("iterstrat not installed, falling back to random split")
        n = len(labels)
        indices = np.random.RandomState(seed).permutation(n)
        split = int(n * (1 - test_size))
        train_idx, test_idx = indices[:split], indices[split:]
    return train_idx, test_idx


def _stratified_split_mcc(labels, test_size=0.2, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    return train_idx, test_idx


def _random_sampling(n, fraction, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    if fraction >= 1.0:
        return indices, np.array([], dtype=int)
    n_labeled = max(min(math.ceil(n * fraction), n - 1), 1)
    return indices[:n_labeled], indices[n_labeled:]


# --- Dataset wrappers ---

class TransformWrapper(Dataset):
    """Wraps a base dataset with SimMIM transform. All samples have labels."""

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        item = self.base[real_idx]
        img_norm, img_raw, mask, non_masked = self.transform(item["image"])
        return img_norm, img_raw, mask, non_masked, item["label"], real_idx


class SemiSupervisedDataset(Dataset):
    """Paired semi-supervised dataset with labeled upsampling.

    Each __getitem__ returns a pair: one unlabeled + one labeled sample.
    Labeled pool is upsampled (with replacement) to match unlabeled size.
    The dataloader then produces batches of 2N images (N unlabeled + N labeled).

    Returns: (x_u, x_raw_u, mask_u, nm_u, y_sentinel, idx_u,
              x_l, x_raw_l, mask_l, nm_l, y_l, idx_l)
    Flattened to 12 elements per sample. The learner concatenates them.
    """

    def __init__(self, base_dataset, labeled_idx, unlabeled_idx,
                 labeled_transform, unlabeled_transform, n_classes, seed=42):
        self.base = base_dataset
        self.labeled_transform = labeled_transform
        self.unlabeled_transform = unlabeled_transform
        self.n_classes = n_classes
        self.unlabeled_idx = unlabeled_idx

        # Upsample labeled to match unlabeled size
        rng = np.random.RandomState(seed)
        self.labeled_idx = labeled_idx[
            rng.choice(len(labeled_idx), size=len(unlabeled_idx), replace=True)
        ]

    def __len__(self):
        return len(self.unlabeled_idx)

    def __getitem__(self, idx):
        # Unlabeled sample
        u_idx = int(self.unlabeled_idx[idx])
        u_item = self.base[u_idx]
        u_norm, u_raw, u_mask, u_nm = self.unlabeled_transform(u_item["image"])
        u_label = torch.full_like(u_item["label"], -1)

        # Matched labeled sample (upsampled)
        l_idx = int(self.labeled_idx[idx])
        l_item = self.base[l_idx]
        l_norm, l_raw, l_mask, l_nm = self.labeled_transform(l_item["image"])
        l_label = l_item["label"]

        return (u_norm, u_raw, u_mask, u_nm, u_label, u_idx,
                l_norm, l_raw, l_mask, l_nm, l_label, l_idx)


# --- DataModule ---

class SSLMAEDataModule(LightningDataModule):
    """Lightning DataModule for SSL-MAE.

    Split logic (respects HuggingFace dataset splits):
      - train + val + test available  -> use as-is
      - train + test available        -> split 10% of train as val
      - train only                    -> split into train 70% / val 10% / test 20%

    Then fraction_labeled is applied to the train split.

    Training modes:
      - "semi_supervised"     : ALL train data used; labeled for both losses, unlabeled for reconstruction only
      - "supervised"          : only labeled fraction used, both losses
      - "supervised_baseline" : only labeled fraction used, classification loss only
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def is_baseline(self):
        return self.config.training.mode == "supervised_baseline"

    def _get_mask_kwargs(self, mask_ratio=None):
        cfg = self.config
        return dict(
            input_size=cfg.data.image_size,
            mask_patch_size=cfg.data.mask_patch_size,
            model_patch_size=cfg.data.model_patch_size,
            mask_ratio=mask_ratio if mask_ratio is not None else cfg.data.mask_ratio,
        )

    def _split_fn(self):
        if self.config.data.learning_task == "mlc":
            return _stratified_split_mlc
        return _stratified_split_mcc

    def setup(self, stage=None):
        cfg = self.config
        seed = cfg.experiment.seed
        split_fn = self._split_fn()

        # Load HF dataset splits
        ds_dict = load_hf_splits(
            hf_dataset_name=cfg.data.get("hf_dataset_name"),
            data_dir=cfg.data.get("data_dir"),
        )
        available = set(ds_dict.keys())
        logger.info(f"Available HF splits: {available}")

        val_key = "validation" if "validation" in available else "val" if "val" in available else None
        has_train = "train" in available
        has_test = "test" in available
        has_val = val_key is not None

        wrap_kwargs = dict(
            learning_task=cfg.data.learning_task,
            n_classes=cfg.data.n_classes,
            image_key=cfg.data.get("image_key", "image"),
            label_key=cfg.data.get("label_key", "label"),
        )

        if has_train and has_val and has_test:
            train_ds = SSLMAEDataset(ds_dict["train"], **wrap_kwargs)
            val_ds = SSLMAEDataset(ds_dict[val_key], **wrap_kwargs, label_to_idx=train_ds.label_to_idx)
            test_ds = SSLMAEDataset(ds_dict["test"], **wrap_kwargs, label_to_idx=train_ds.label_to_idx)
            train_idx = np.arange(len(train_ds))
            val_idx = np.arange(len(val_ds))
            test_idx = np.arange(len(test_ds))
            logger.info("Using existing train/val/test splits from dataset")

        elif has_train and has_test:
            train_ds = SSLMAEDataset(ds_dict["train"], **wrap_kwargs)
            test_ds = SSLMAEDataset(ds_dict["test"], **wrap_kwargs, label_to_idx=train_ds.label_to_idx)
            train_labels = train_ds.get_all_labels()
            train_sub_idx, val_sub_idx = split_fn(train_labels, test_size=0.1, seed=seed)
            train_idx = train_sub_idx
            val_idx = val_sub_idx
            val_ds = train_ds
            test_idx = np.arange(len(test_ds))
            logger.info("Using existing train/test splits; split 10%% of train as val")

        elif has_train:
            train_ds = SSLMAEDataset(ds_dict["train"], **wrap_kwargs)
            all_labels = train_ds.get_all_labels()
            trainval_idx, test_idx = split_fn(all_labels, test_size=0.2, seed=seed)
            trainval_labels = all_labels[trainval_idx]
            train_sub_idx, val_sub_idx = split_fn(trainval_labels, test_size=0.125, seed=seed)
            train_idx = trainval_idx[train_sub_idx]
            val_idx = trainval_idx[val_sub_idx]
            val_ds = train_ds
            test_ds = train_ds
            logger.info("Only train split available; split into 70%%/10%%/20%%")

        else:
            raise ValueError(f"Dataset has no 'train' split. Available: {available}")

        # Apply fraction_labeled
        labeled_sub_idx, unlabeled_sub_idx = _random_sampling(
            len(train_idx), cfg.data.fraction_labeled, seed=seed
        )
        labeled_idx = train_idx[labeled_sub_idx]
        unlabeled_idx = train_idx[unlabeled_sub_idx]

        # Setup transforms
        mask_kwargs = self._get_mask_kwargs()
        train_transform = SimMIMTransform('labeled', **mask_kwargs)
        unlabeled_transform = SimMIMTransform('unlabeled', **mask_kwargs)
        val_transform = SimMIMTransform('val', **self._get_mask_kwargs(mask_ratio=0.0))

        # Setup train dataset based on mode
        if cfg.training.mode == "semi_supervised" and len(unlabeled_idx) > 0:
            self.train_dataset = SemiSupervisedDataset(
                train_ds, labeled_idx, unlabeled_idx,
                train_transform, unlabeled_transform, cfg.data.n_classes, seed=seed
            )
            logger.info(f"Semi-supervised: {len(labeled_idx)} labeled (upsampled to {len(unlabeled_idx)}) "
                        f"+ {len(unlabeled_idx)} unlabeled = {len(self.train_dataset)} paired samples")
        else:
            # supervised and supervised_baseline: only labeled data
            self.train_dataset = TransformWrapper(train_ds, labeled_idx, train_transform)
            logger.info(f"{cfg.training.mode}: {len(labeled_idx)} labeled samples")

        self.val_dataset = TransformWrapper(val_ds, val_idx, val_transform)
        self.test_dataset = TransformWrapper(test_ds, test_idx, val_transform)

        logger.info(f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def train_dataloader(self):
        n = len(self.train_dataset)
        batch_size = min(self.config.data.batch_size, n)
        if batch_size < self.config.data.batch_size:
            logger.warning(f"Train set ({n} samples) smaller than batch_size "
                           f"({self.config.data.batch_size}), using batch_size={batch_size}")
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.data.num_workers,
            drop_last=n > batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.config.data.batch_size,
            shuffle=False, pin_memory=True,
            num_workers=self.config.data.num_workers, drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.data.batch_size,
            shuffle=False, pin_memory=True,
            num_workers=self.config.data.num_workers, drop_last=False,
        )
