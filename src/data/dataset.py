import logging
from io import BytesIO

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_hf_splits(hf_dataset_name=None, data_dir=None, cache_dir=None):
    """Load a HuggingFace dataset and return the DatasetDict with available splits.

    Returns:
        DatasetDict with keys like 'train', 'test', 'validation', etc.
    """
    if hf_dataset_name is not None:
        ds = load_dataset(hf_dataset_name, cache_dir=cache_dir, trust_remote_code=True)
    elif data_dir is not None:
        if data_dir.endswith('.csv'):
            ds = load_dataset("csv", data_files=data_dir, cache_dir=cache_dir)
        else:
            ds = load_dataset("imagefolder", data_dir=data_dir, cache_dir=cache_dir)
    else:
        raise ValueError("Must provide either hf_dataset_name or data_dir")

    # Wrap single dataset in a DatasetDict
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    return ds


class SSLMAEDataset(Dataset):
    """HuggingFace dataset wrapper for SSL-MAE.

    Wraps a single HF dataset split (not a DatasetDict).
    Handles both MLC (multi-label) and MCC (multi-class) tasks.

    MLC labels can be either:
    - Multi-hot vectors: [0, 0, 1, 0, 1, ...] (fixed length = n_classes)
    - Lists of class indices: [7, 15] (variable length, converted to multi-hot)
    """

    def __init__(self, hf_dataset, learning_task="mlc", n_classes=None,
                 image_key="image", label_key="label", label_to_idx=None):
        super().__init__()
        self.dataset = hf_dataset
        self.learning_task = learning_task
        self.n_classes = n_classes
        self.label_to_idx = label_to_idx
        self.mlc_is_indices = False  # True if MLC labels are class index lists
        self._resolve_columns(image_key, label_key)

        # Build label encoding if not provided
        if self.label_to_idx is None:
            self._build_label_encoding()

        # Detect MLC label format
        if self.learning_task == "mlc":
            self._detect_mlc_format()

    def _resolve_columns(self, image_key, label_key):
        """Auto-detect image and label column names."""
        columns = self.dataset.column_names
        for candidates, attr in [
            ([image_key, "image", "img", "pixel_values"], "image_col"),
            ([label_key, "label", "labels", "target", "class"], "label_col"),
        ]:
            resolved = None
            for col in candidates:
                if col in columns:
                    resolved = col
                    break
            if resolved is None:
                resolved = columns[0] if attr == "image_col" else columns[-1]
                logger.warning(f"Column not found, using fallback: {resolved}")
            setattr(self, attr, resolved)

    def _build_label_encoding(self):
        """Build string-to-int mapping if labels are strings (MCC)."""
        sample_label = self.dataset[0][self.label_col]
        if isinstance(sample_label, str):
            unique_labels = sorted(set(self.dataset[self.label_col]))
            self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    def _detect_mlc_format(self):
        """Detect whether MLC labels are multi-hot vectors or class index lists."""
        sample = self.dataset[0][self.label_col]
        if not isinstance(sample, (list, np.ndarray)):
            return

        # Check if all values are 0/1 and length matches n_classes -> multi-hot
        is_binary = all(v in (0, 1, 0.0, 1.0) for v in sample)
        if is_binary and self.n_classes is not None and len(sample) == self.n_classes:
            self.mlc_is_indices = False
            logger.info("MLC labels detected as multi-hot vectors")
        else:
            # Variable-length lists of class indices
            self.mlc_is_indices = True
            if self.n_classes is None:
                # Infer n_classes from max index
                all_labels = self.dataset[self.label_col]
                self.n_classes = max(max(l) for l in all_labels if len(l) > 0) + 1
                logger.info(f"Inferred n_classes={self.n_classes} from label indices")
            logger.info("MLC labels detected as class index lists, will convert to multi-hot")

    def _process_image(self, raw_image):
        """Convert various image formats to RGB PIL Image."""
        if isinstance(raw_image, Image.Image):
            return raw_image.convert("RGB")
        elif isinstance(raw_image, np.ndarray):
            return Image.fromarray(raw_image).convert("RGB")
        elif isinstance(raw_image, bytes):
            return Image.open(BytesIO(raw_image)).convert("RGB")
        elif isinstance(raw_image, dict) and "bytes" in raw_image:
            return Image.open(BytesIO(raw_image["bytes"])).convert("RGB")
        else:
            return Image.fromarray(np.array(raw_image)).convert("RGB")

    def _process_label(self, raw_label):
        """Convert label to tensor."""
        if self.learning_task == "mlc":
            if self.mlc_is_indices:
                # Convert list of class indices to multi-hot vector
                multi_hot = torch.zeros(self.n_classes, dtype=torch.float)
                for idx in raw_label:
                    multi_hot[idx] = 1.0
                return multi_hot
            return torch.tensor(raw_label, dtype=torch.float)
        else:
            if self.label_to_idx is not None:
                return torch.tensor(self.label_to_idx[raw_label], dtype=torch.long)
            return torch.tensor(raw_label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self._process_image(sample[self.image_col])
        label = self._process_label(sample[self.label_col])
        return {"image": image, "label": label}

    def get_all_labels(self):
        """Extract all labels as a numpy array (for stratified splitting)."""
        labels = self.dataset[self.label_col]
        if self.learning_task == "mlc":
            if self.mlc_is_indices:
                # Convert all to multi-hot
                n = len(labels)
                multi_hot = np.zeros((n, self.n_classes), dtype=np.float32)
                for i, label_list in enumerate(labels):
                    for idx in label_list:
                        multi_hot[i, idx] = 1.0
                return multi_hot
            return np.array(labels, dtype=np.float32)
        else:
            if self.label_to_idx is not None:
                return np.array([self.label_to_idx[l] for l in labels])
            return np.array(labels)
