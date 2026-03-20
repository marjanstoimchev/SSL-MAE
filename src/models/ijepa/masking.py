"""Block-based masking for I-JEPA (from Meta's implementation).

Generates context and target block masks:
- Context: 1 large block (85-100% of patches) with target regions removed
- Target: N smaller blocks (15-20% each) to predict in latent space

Masks are indices of patches to KEEP (not binary masks).
"""

import math
import torch


class BlockMaskGenerator:
    """Generate I-JEPA style block masks.

    Returns (context_mask, target_masks, binary_mask, non_masked):
    - context_mask: (num_context_patches,) indices of context patches
    - target_masks: list of (num_target_patches,) indices per target block
    - binary_mask: (num_patches,) binary mask compatible with MAE interface (1=masked)
    - non_masked: (num_visible,) indices of visible patches (same as context_mask)
    """

    def __init__(self, input_size=224, patch_size=16,
                 enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
                 aspect_ratio=(0.75, 1.5), npred=4, min_keep=10,
                 allow_overlap=False):
        self.height = input_size // patch_size
        self.width = input_size // patch_size
        self.num_patches = self.height * self.width
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    def _sample_block_size(self, scale, aspect_ratio_scale):
        rand_val = torch.rand(1).item()
        min_s, max_s = scale
        mask_scale = min_s + rand_val * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        min_ar, max_ar = aspect_ratio_scale
        ar = min_ar + rand_val * (max_ar - min_ar)

        h = int(round(math.sqrt(max_keep * ar)))
        w = int(round(math.sqrt(max_keep / ar)))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        return h, w

    def _sample_block(self, h, w, acceptable_regions=None):
        """Sample a rectangular block, return (indices, complement_mask)."""
        for _ in range(100):
            top = torch.randint(0, max(self.height - h, 1), (1,)).item()
            left = torch.randint(0, max(self.width - w, 1), (1,)).item()

            mask = torch.zeros(self.height, self.width, dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1

            if acceptable_regions is not None:
                for region in acceptable_regions:
                    mask = mask * region

            indices = torch.nonzero(mask.flatten()).squeeze(-1)
            if len(indices) >= self.min_keep:
                complement = torch.ones(self.height, self.width, dtype=torch.int32)
                complement[top:top+h, left:left+w] = 0
                return indices, complement

        # Fallback: random patches
        indices = torch.randperm(self.num_patches)[:max(self.min_keep, h * w)]
        complement = torch.ones(self.height, self.width, dtype=torch.int32)
        return indices, complement

    def __call__(self):
        """Generate context and target masks.

        Returns:
            context_indices: (N_ctx,) indices of context patches
            target_indices: (N_tgt_total,) concatenated indices of all target patches
            binary_mask: (num_patches,) binary mask (1=masked/target, 0=visible/context)
            non_masked: (N_ctx,) same as context_indices (for interface compatibility)
        """
        # Sample target blocks
        p_size = self._sample_block_size(self.pred_mask_scale, self.aspect_ratio)
        target_indices_list = []
        complements = []
        for _ in range(self.npred):
            indices, complement = self._sample_block(p_size[0], p_size[1])
            target_indices_list.append(indices)
            complements.append(complement)

        # Sample context block (avoiding target regions unless allow_overlap)
        e_size = self._sample_block_size(self.enc_mask_scale, (1., 1.))
        acceptable = complements if not self.allow_overlap else None
        context_indices, _ = self._sample_block(e_size[0], e_size[1],
                                                 acceptable_regions=acceptable)

        # Concatenate all target indices (deduplicated)
        all_target = torch.cat(target_indices_list).unique()

        # Build binary mask: 1 where target, 0 where context
        binary_mask = torch.zeros(self.num_patches, dtype=torch.long)
        binary_mask[all_target] = 1

        # non_masked = context indices (visible patches)
        non_masked = context_indices

        return context_indices, all_target, binary_mask, non_masked
