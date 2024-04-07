import logging
import torch.nn as nn
from collections import defaultdict
from .losses import MaskedLoss
from .layers import W, Decoder, Classifier
from .encoders import DeiTEncoder, ViTEncoder
from utils.model_utils import random_masking

class SSLMAE(nn.Module):
    def __init__(self,
                 encoder,
                 learning_task,
                 n_classes,
                 w = None,
                 ):
        
        super().__init__()
            
        self.encoder = encoder

        if learning_task == "mlc":
            self.c_loss = nn.BCEWithLogitsLoss()
        elif learning_task == "mcc":
            self.c_loss = nn.CrossEntropyLoss()

        encoder_config = self.encoder.config

        # definition of the decoder
        self.decoder = Decoder(
            encoder_config.hidden_size, 
            encoder_config.encoder_stride, 
            encoder_config.num_channels
            )

        # definition of the classifier
        self.classifier = Classifier(encoder_config.hidden_size, n_classes)

        # losses
        self.m_loss = MaskedLoss(encoder_config.image_size, encoder_config.patch_size)

        self.hidden_size = encoder_config.hidden_size
        self.patch_size = encoder_config.patch_size
        self.w_ = W() if w is None else w
        self.w = w

    def forward_w(self):
        w = self.w() if self.w is None else self.w
        return w

    def forward_mask(self, x, mask_ratio):
        """
        x: [batch_size, c, h, w], image
        mask_ratio: float \in[0, 1]
        """
        x_masked, mask, ids_keep, ids_restore = random_masking(
            x, 
            patch_size = self.patch_size,
            mask_ratio = mask_ratio, 
            mode = "transformer"
            )
        return x_masked, mask, ids_keep, ids_restore

    def forward_classify(self, embeddings, labels):
        logits = self.classifier(embeddings)
        labeled_logits = logits[:len(labels)] # get only labeled logits
        c_loss =  self.c_loss(labeled_logits, labels) 
        return c_loss

    def forward(self, imgs, labels = None, mask_ratio = 0.75):
        outputs = defaultdict(lambda: None)
        _, mask, ids_keep, _ = self.forward_mask(imgs, mask_ratio = mask_ratio)
        sequence_output, embeddings = self.encoder(imgs, bool_masked_pos=mask.bool(), ids_keep = ids_keep)
        reconstructed_pixel_values = self.decoder(sequence_output)
        r_loss = self.m_loss(imgs, reconstructed_pixel_values, bool_masked_pos = mask)

        if labels is not None:
            c_loss = self.forward_classify(embeddings, labels)
            outputs['c_loss'] = c_loss

        outputs['r_loss'] = r_loss 
        outputs['reconstruction'] = reconstructed_pixel_values
        outputs['mask'] = mask
        return outputs
    

def ssl_mae_deit(model_size, learning_task, n_classes, w):
    """
    Initializes a model based on the DeiT architecture with SSL MAE.

    Parameters:
    - model_size (str): Size of the model ('base', 'large').
    - learning_task (str): The learning task for the model.
    - n_classes (int): The number of classes for classification tasks.
    - w (float): Specific weight parameter for model configuration.

    Returns:
    - An instance of the configured model.
    """
    SUPPORTED_MODEL_SIZES = ['tiny', 'small', 'base']
    if model_size not in SUPPORTED_MODEL_SIZES:
        logging.error(f"Unsupported model size: {model_size}. Supported sizes: {SUPPORTED_MODEL_SIZES}")
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {SUPPORTED_MODEL_SIZES}")

    encoder = ViTEncoder.from_pretrained(f'google/vit-{model_size}-patch16-224-in21k')

    model = SSLMAE(
        encoder=encoder,
        learning_task=learning_task,
        n_classes=n_classes,
        w=w,
    )
    return model

def ssl_mae_vit(model_size, learning_task, n_classes, w):
    """
    Initializes a model based on the ViT architecture with SSL MAE.

    Parameters:
    - model_size (str): Size of the model ('base', 'large').
    - learning_task (str): The learning task for the model.
    - n_classes (int): The number of classes for classification tasks.
    - w (float): Specific weight parameter for model configuration.

    Returns:
    - An instance of the configured model.
    """
    SUPPORTED_MODEL_SIZES = ['base', 'large']
    if model_size not in SUPPORTED_MODEL_SIZES:
        logging.error(f"Unsupported model size: {model_size}. Supported sizes: {SUPPORTED_MODEL_SIZES}")
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {SUPPORTED_MODEL_SIZES}")

    encoder = DeiTEncoder.from_pretrained(f'facebook/deit-{model_size}-distilled-patch16-224')

    model = SSLMAE(
        encoder=encoder,
        learning_task=learning_task,
        n_classes=n_classes,
        w=w,
    )
    return model
