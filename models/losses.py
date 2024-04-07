import torch.nn as nn

class MaskedLoss(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = 3

    def forward(self, pixel_values, pred, bool_masked_pos):   
        if bool_masked_pos.sum() != 0:
            size = self.image_size // self.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.patch_size, 1)
                .repeat_interleave(self.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, pred, reduction="none")
            loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.num_channels
        else:
            loss = nn.functional.l1_loss(pixel_values, pred, reduction="mean")
        return loss
    
