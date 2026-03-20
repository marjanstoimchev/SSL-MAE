"""SSL model implementations: MAE and I-JEPA."""

from .mae import ssl_mae, SSLMAE
from .ijepa import ssl_ijepa, SSLIJEPA
from .common import W, Classifier

__all__ = ['ssl_mae', 'SSLMAE', 'ssl_ijepa', 'SSLIJEPA', 'W', 'Classifier']
