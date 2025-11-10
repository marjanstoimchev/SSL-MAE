"""Dataset implementations for various remote sensing datasets."""

from .dataset_selector import DatasetSelectorMLC
from .ucm_dataset import *
from .aid_dataset import *
from .mlrsnet_dataset import *
from .ben_dataset import *
from .ankara_dataset import *
from .dfc_15_dataset import *

__all__ = ['DatasetSelectorMLC']
