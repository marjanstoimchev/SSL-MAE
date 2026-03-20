import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import VisionTransformer, Block
from ..common.layers import W
from ..common.registry import load_pretrained_vit


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """2D sincos positional embedding (from I-JEPA / MAE)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1].reshape(-1))
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.)
    omega = 1. / 10000 ** omega
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def apply_masks(x, masks):
    """Gather patches by index. masks: list of (B, M) index tensors."""
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


class Predictor(nn.Module):
    """Narrow ViT predictor for I-JEPA.

    Takes context encoder output + positional info for target patches,
    predicts target representations in latent space.
    """

    def __init__(self, num_patches, embed_dim, predictor_embed_dim=384,
                 depth=6, num_heads=12):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        grid_size = int(num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, grid_size)
        self.predictor_pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False
        )

        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

    def forward(self, x, masks_ctx, masks_tgt):
        """Predict target representations from context.

        Args:
            x: (B, N_ctx, embed_dim) context encoder output (only context patches)
            masks_ctx: list of (B, N_ctx) index tensors for context patches
            masks_tgt: list of (B, N_tgt) index tensors for target patches

        Returns:
            (B*n_tgt, N_tgt, embed_dim) predicted target representations
        """
        B = x.size(0)

        # Project to predictor dimension
        x = self.predictor_embed(x)

        # Add positional embeddings to context tokens
        pos_embed = self.predictor_pos_embed.expand(B, -1, -1)
        x += apply_masks(pos_embed, masks_ctx)

        # Create prediction tokens with target positional embeddings
        pos_tgt = apply_masks(pos_embed, masks_tgt)
        pred_tokens = self.mask_token.repeat(pos_tgt.size(0), pos_tgt.size(1), 1) + pos_tgt

        # Repeat context for each target mask
        x = x.repeat(len(masks_tgt), 1, 1)
        N_ctx = x.size(1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Extract prediction tokens and project back
        x = x[:, N_ctx:]
        x = self.predictor_proj(x)
        return x


class IJEPAEncoder(VisionTransformer):
    """I-JEPA context encoder. Processes only visible (context) patches."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.head

    def forward(self, x, masks=None):
        """Forward pass. If masks provided, only keeps indexed patches.

        Args:
            x: (B, 3, H, W) images
            masks: list of (B, M) index tensors. If None, processes all patches.

        Returns:
            (B, N_kept, embed_dim) patch representations
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        # Keep only context patches (unlike MAE which replaces with mask tokens)
        if masks is not None:
            x = x[:, self.num_prefix_tokens:, :]  # remove CLS before masking
            x = apply_masks(x, masks)
        else:
            x = x[:, self.num_prefix_tokens:, :]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class SSLIJEPA(nn.Module):
    """SSL-IJEPA: Joint-Embedding Predictive Architecture for semi-supervised learning.

    Components:
    - Context encoder (ViT): processes visible patches, receives gradients
    - Target encoder (ViT): processes full image, EMA updated, no gradients
    - Predictor (narrow ViT): predicts target representations from context
    - Classifier: MLP head on context features for classification

    Interface compatible with SSLMAE for the training pipeline.
    """

    def __init__(self, encoder, predictor, criterion, n_classes=17, w=None, ema_decay=0.996):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.hidden_dim = encoder.embed_dim
        self.embed_dim = encoder.embed_dim

        # Target encoder (EMA of context encoder)
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.requires_grad_(False)

        # Classifier head (same as MAE)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, n_classes),
        )

        self.criterion = criterion
        self.ema_decay = ema_decay

        trainable = w is None
        fixed_value = 0.5 if w is None else w
        self.w = W(trainable=trainable, fixed_value=fixed_value)

    def forward_features(self, x, mask=None):
        """Context encoder forward. For interface compatibility with learner.

        When mask is a binary mask (MAE-style), converts to index mask.
        When mask is None, processes all patches.
        """
        if mask is not None:
            # Convert binary mask (1=masked) to context indices (0=visible)
            non_masked_bool = ~mask.bool()
            # Get indices of visible patches
            ctx_indices = [torch.where(non_masked_bool[i])[0] for i in range(mask.size(0))]
            # Pad to same length
            max_len = max(len(c) for c in ctx_indices)
            ctx_padded = torch.stack([
                F.pad(c, (0, max_len - len(c)), value=0) for c in ctx_indices
            ]).to(x.device)
            return self.encoder(x, [ctx_padded])
        return self.encoder(x)

    @torch.no_grad()
    def forward_target(self, x):
        """Target encoder: full image, no masking, no gradients."""
        z = self.target_encoder(x)  # all patches
        return F.layer_norm(z, (z.size(-1),))

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder from context encoder."""
        for param_t, param_c in zip(self.target_encoder.parameters(),
                                     self.encoder.parameters()):
            param_t.data.mul_(self.ema_decay).add_(param_c.data, alpha=1 - self.ema_decay)

    def forward_ssl(self, x, masks_ctx, masks_tgt):
        """I-JEPA self-supervised forward.

        Args:
            x: (B, 3, H, W) images
            masks_ctx: list of (B, N_ctx) context index tensors
            masks_tgt: list of (B, N_tgt) target index tensors

        Returns:
            r_loss: smooth L1 in representation space
        """
        B = x.size(0)

        # Target representations (no grad)
        h = self.forward_target(x)
        h = apply_masks(h, masks_tgt)
        # Repeat for number of context masks
        from src.models.ijepa.utils import repeat_interleave_batch
        h = repeat_interleave_batch(h, B, repeat=len(masks_ctx))

        # Context encoder + predictor
        z = self.encoder(x, masks_ctx)
        z = self.predictor(z, masks_ctx, masks_tgt)

        # Loss in representation space
        r_loss = F.smooth_l1_loss(z, h)
        return r_loss

    def forward_classifier_from_features(self, z, labels, labeled_mask):
        """Classify from encoder features."""
        x = z[labeled_mask]
        y = labels[labeled_mask]
        x = x.mean(1)
        logits = self.fc(x)
        loss = self.criterion(logits, y)
        return logits, loss

    def forward(self, x, x_raw, mask, non_masked=None, labels=None):
        """Forward pass compatible with SSLMAE interface.

        Uses binary mask to derive context/target indices for I-JEPA.
        """
        outputs = {}
        B = x.size(0)

        # Convert binary mask to I-JEPA index masks
        mask_bool = mask.bool()
        ctx_indices = torch.stack([torch.where(~mask_bool[i])[0] for i in range(B)])
        tgt_indices = torch.stack([torch.where(mask_bool[i])[0] for i in range(B)])
        masks_ctx = [ctx_indices.to(x.device)]
        masks_tgt = [tgt_indices.to(x.device)]

        # Self-supervised loss
        r_loss = self.forward_ssl(x, masks_ctx, masks_tgt)
        outputs['r_loss'] = r_loss
        outputs['mask'] = mask

        # Classification on visible patches of labeled images
        c_loss = None
        if labels is not None:
            labeled_mask = (labels >= 0).any(dim=-1) if labels.dim() > 1 else labels >= 0
            if labeled_mask.any():
                # Get context features for classification
                z_ctx = self.encoder(x, masks_ctx)
                logits, c_loss = self.forward_classifier_from_features(
                    z_ctx, labels, labeled_mask
                )
                outputs['logits'] = logits
                outputs['c_loss'] = c_loss

        w = self.w()
        if c_loss is None:
            c_loss = 0
        outputs['c_loss'] = outputs.get('c_loss', c_loss)
        outputs['loss'] = (1 - w) * r_loss + w * c_loss
        outputs['w'] = w

        # EMA update during training only
        if self.training:
            self.update_target_encoder()

        return outputs


def ssl_ijepa(architecture, model_size, learning_task, n_classes, w,
              predictor_embed_dim=384, predictor_depth=6, ema_decay=0.996):
    """Create an SSL-IJEPA model with a pretrained timm ViT backbone."""
    import torch

    criterions = {"mcc": nn.CrossEntropyLoss(), "mlc": nn.BCEWithLogitsLoss()}
    model_name, pretrained_state, kwargs = load_pretrained_vit(architecture, model_size)

    # Create encoder
    encoder = IJEPAEncoder(**kwargs)

    # Handle pos_embed size mismatch
    if pretrained_state['pos_embed'].shape != encoder.pos_embed.shape:
        pretrained_pos = pretrained_state['pos_embed']
        model_n = encoder.pos_embed.shape[1]
        pretrained_state['pos_embed'] = torch.cat([
            pretrained_pos[:, :1, :],
            pretrained_pos[:, -model_n + 1:, :]
        ], dim=1)

    # Load pretrained weights into encoder
    encoder_keys = set(encoder.state_dict().keys())
    encoder_state = {k: v for k, v in pretrained_state.items() if k in encoder_keys}
    msg = encoder.load_state_dict(encoder_state, strict=False)
    print(f"[I-JEPA] Loaded pretrained encoder from '{model_name}'")

    # Create predictor
    predictor = Predictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=kwargs['num_heads'],
    )

    # Assemble model
    model = SSLIJEPA(
        encoder=encoder,
        predictor=predictor,
        criterion=criterions[learning_task],
        n_classes=n_classes,
        w=w,
        ema_decay=ema_decay,
    )

    model.train()
    return model
