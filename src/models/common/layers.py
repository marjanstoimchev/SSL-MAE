import torch
import torch.nn as nn


class W(nn.Module):
    """Adaptive weight in [0, 1] for balancing reconstruction and classification losses.

    - Learnable: single scalar parameter, sigmoid-constrained to [0, 1].
      Initialized so sigmoid(raw) = fixed_value (default 0.5).
    - Fixed: constant value, not optimized.
    """

    def __init__(self, trainable=True, fixed_value=0.5):
        super().__init__()
        self.trainable = trainable
        if trainable:
            raw = torch.log(torch.tensor(fixed_value / (1 - fixed_value)))
            self.w_param = nn.Parameter(raw)
        else:
            self.register_buffer('w_param', torch.tensor(fixed_value))

    def forward(self):
        if self.trainable:
            return torch.sigmoid(self.w_param)
        return self.w_param


class Classifier(nn.Module):
    """Simple linear classification head."""

    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.head(x)
