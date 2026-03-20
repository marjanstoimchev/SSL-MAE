import numpy as np
import torch
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MaskGenerator:
    """Generates boolean masks for the SimMIM pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where 1 = masked.
    """

    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.75):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        all_indices = np.arange(self.token_count)
        non_mask_idx = np.isin(all_indices, mask_idx, invert=True)
        non_mask_idx = all_indices[non_mask_idx]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten()), torch.tensor(non_mask_idx)


class SimMIMTransform:
    """Image transform with mask generation for SimMIM-style training.

    Returns (img_normalized, img_raw, mask, non_masked):
    - img_normalized: ImageNet-normalized tensor for the encoder
    - img_raw: [0,1] tensor for reconstruction target
    - mask, non_masked: from MaskGenerator

    data_type controls augmentation strength:
    - 'labeled': standard augmentations (crop, flip)
    - 'unlabeled': heavy augmentations (crop, flip, color jitter, grayscale, blur)
    - 'val'/'test': no augmentation, just resize
    """

    def __init__(self, data_type, **kwargs):
        size = kwargs['input_size']
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if data_type == 'unlabeled':
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.RandomResizedCrop(size, scale=(0.4, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                T.Resize((size, size)),
                T.ToTensor(),
            ])
        elif data_type in ['labeled', 'both']:
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.RandomResizedCrop(size, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.Resize((size, size)),
                T.ToTensor(),
            ])
        elif data_type in ['val', 'test']:
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((size, size)),
                T.ToTensor(),
            ])

        self.mask_generator = MaskGenerator(**kwargs)

    def __call__(self, img):
        img_raw = self.transform_img(img)        # [0, 1] for reconstruction target
        img_norm = self.normalize(img_raw)        # ImageNet-normalized for encoder
        mask, non_masked = self.mask_generator()
        return img_norm, img_raw, mask, non_masked
