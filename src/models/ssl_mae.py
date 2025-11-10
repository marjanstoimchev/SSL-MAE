import logging
import torch
import torch.nn as nn
from .layers import Classifier, W
from collections import defaultdict
from transformers import AutoModelForMaskedImageModeling

class SSLMAE(nn.Module):
    def __init__(self, backbone, criterion, n_classes = 17, w = None):
        super().__init__()

        self.hidden_dim = backbone.config.hidden_size
        self.patch_size = backbone.config.patch_size

        self.backbone = backbone
        self.fc = Classifier(self.hidden_dim, n_classes)

        num_channels = 3
        decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.patch_size**2 * num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.patch_size),
            nn.Sigmoid()
        )

        self.criterion = criterion
        self.backbone.decoder = decoder

        trainable = w is None
        fixed_value = 0.5 if w is None else w  # Assuming default value is 0.5 if `w` is None
        self.w = W(trainable=trainable, fixed_value=fixed_value)

    def forward_w(self):
        return self.w()
    
    def forward_classifier(self, x, non_masked, labels):
        x = x[:len(labels)] # get only labeled images
        # get only the visible latent representations
        if non_masked is not None:
            non_masked = non_masked.unsqueeze(-1).expand(-1, -1, self.hidden_dim) 
            x = torch.gather(x, 1, non_masked)

        x = x.mean(1)
        logits = self.fc(x)
        loss = self.criterion(logits, labels)
        return logits, loss
    
    def forward(self, x, mask, non_masked = None, labels = None):
        
        outputs = defaultdict(lambda: None)
        outputs_ = self.backbone(x, mask, output_hidden_states = True)
        z = outputs_.hidden_states[-1][:, 1:-1]

        r_loss = outputs_.loss
        reconstruction = outputs_.reconstruction
        outputs['x_rec'] = reconstruction
        outputs['mask'] = mask

        w = self.forward_w()
        
        if labels is not None:
            logits, c_loss = self.forward_classifier(z, non_masked, labels.float())
            outputs['logits'] = logits
        else:
            c_loss = 0

        loss = (1-w)*r_loss + w*c_loss
        outputs['loss'] = loss
        outputs['w'] = w
        return outputs
    
def ssl_mae(architecture, model_size, learning_task, n_classes, w):
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

    model_config = {
            "deit": {
                "model_names": {
                    "tiny": "facebook/deit-base-distilled-patch16-224",
                    "small": "facebook/deit-small-distilled-patch16-224",
                    "base": "facebook/deit-base-distilled-patch16-224",
                }
            },
            "vit": {
                "model_names": {
                    "base": "google/vit-base-patch16-224-in21k",
                    "large": "google/vit-large-patch16-224-in21k",
                }
            }
        }

    backbone = AutoModelForMaskedImageModeling.from_pretrained(model_config[architecture]['model_names'][model_size])

    criterions = {
        "mcc": nn.CrossEntropyLoss(),
        "mlc": nn.BCEWithLogitsLoss()
        }

    model = SSLMAE(
        backbone, 
        criterions[learning_task],
        n_classes=n_classes,
        w=w
        )

    return model
