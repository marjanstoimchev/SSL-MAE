import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from ..common.layers import Classifier, W
from ..common.registry import load_pretrained_vit


class SSLMAE(VisionTransformer):
    """SSL-MAE: Vision Transformer with masked reconstruction and adaptive joint learning.

    Inherits from timm's VisionTransformer, adding:
    - Learnable mask token (SimMIM-style)
    - Lightweight pixel decoder for reconstruction
    - MLP classifier head on visible patches
    - Adaptive weight w balancing reconstruction and classification
    - Patch-normalized reconstruction targets (MSE, from MAE)
    """

    def __init__(self, criterion, n_classes=17, w=None, **kwargs):
        super().__init__(**kwargs)

        del self.head

        self._patch_size = self.patch_embed.patch_size[0]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, self._patch_size ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self._patch_size),
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, n_classes),
        )

        self.criterion = criterion
        self.hidden_dim = self.embed_dim

        trainable = w is None
        fixed_value = 0.5 if w is None else w
        self.w = W(trainable=trainable, fixed_value=fixed_value)

    def patchify(self, x):
        p = self._patch_size
        B, C, H, W = x.shape
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x):
        p = self._patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(-1, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, h * p, w * p)
        return x

    def denormalize_reconstruction(self, x_rec, x_raw):
        target = self.patchify(x_raw)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred = self.patchify(x_rec)
        pred = pred * (var.sqrt() + 1e-6) + mean
        return self.unpatchify(pred).clamp(0, 1)

    def forward_features(self, x, mask=None):
        x = self.patch_embed(x)
        if mask is not None:
            B, N, D = x.shape
            mask_bool = mask.bool().unsqueeze(-1)
            x = torch.where(mask_bool, self.mask_token.expand(B, N, D), x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, self.num_prefix_tokens:, :]

    def forward_decoder(self, z, x_original, mask):
        B, N, D = z.shape
        h = w = int(N ** 0.5)
        z_spatial = z.transpose(1, 2).reshape(B, D, h, w)
        x_rec = self.decoder(z_spatial)

        target = self.patchify(x_original)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var.sqrt() + 1e-6)
        pred = self.patchify(x_rec)

        mask_flat = mask.float()
        loss = (F.mse_loss(pred, target, reduction='none').mean(dim=-1) * mask_flat).sum()
        loss = loss / (mask_flat.sum() + 1e-8)
        return loss, x_rec

    def forward_classifier(self, z, non_masked, labels, labeled_mask):
        x = z[labeled_mask]
        nm = non_masked[labeled_mask]
        y = labels[labeled_mask]
        if nm is not None:
            nm_exp = nm.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            x = torch.gather(x, 1, nm_exp)
        x = x.mean(1)
        logits = self.fc(x)
        loss = self.criterion(logits, y)
        return logits, loss

    def forward(self, x, x_raw, mask, non_masked=None, labels=None):
        outputs = {}
        z = self.forward_features(x, mask)

        r_loss, x_rec = self.forward_decoder(z, x_raw, mask)
        outputs['x_rec'] = x_rec
        outputs['mask'] = mask
        outputs['r_loss'] = r_loss

        c_loss = None
        if labels is not None:
            labeled_mask = (labels >= 0).any(dim=-1) if labels.dim() > 1 else labels >= 0
            if labeled_mask.any():
                logits, c_loss = self.forward_classifier(z, non_masked, labels, labeled_mask)
                outputs['logits'] = logits
                outputs['c_loss'] = c_loss

        w = self.w()
        if c_loss is None:
            c_loss = 0
        outputs['c_loss'] = outputs.get('c_loss', c_loss)
        outputs['loss'] = (1 - w) * r_loss + w * c_loss
        outputs['w'] = w
        return outputs


def ssl_mae(architecture, model_size, learning_task, n_classes, w):
    """Create an SSL-MAE model with a pretrained timm ViT backbone."""
    import torch

    criterions = {"mcc": nn.CrossEntropyLoss(), "mlc": nn.BCEWithLogitsLoss()}
    model_name, pretrained_state, kwargs = load_pretrained_vit(architecture, model_size)

    model = SSLMAE(criterion=criterions[learning_task], n_classes=n_classes, w=w, **kwargs)

    # Handle pos_embed size mismatch (distilled models have extra distill token)
    if pretrained_state['pos_embed'].shape != model.pos_embed.shape:
        pretrained_pos = pretrained_state['pos_embed']
        model_n = model.pos_embed.shape[1]
        pretrained_state['pos_embed'] = torch.cat([
            pretrained_pos[:, :1, :],
            pretrained_pos[:, -model_n + 1:, :]
        ], dim=1)

    model_keys = set(model.state_dict().keys())
    pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_keys}

    msg = model.load_state_dict(pretrained_state, strict=False)
    print(f"[MAE] Loaded pretrained weights from '{model_name}'")
    if msg.missing_keys:
        print(f"  New parameters: {msg.missing_keys}")

    model.train()
    return model
