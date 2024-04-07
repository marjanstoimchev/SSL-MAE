import math
import torch
import torch.nn as nn

class W(nn.Module):
    def __init__(self):
        super(W, self).__init__()
        self.w_param = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        torch.nn.init.constant_(self.w_param, 1.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        return self.sigmoid(self.w_param)
    
class Classifier(nn.Module):
    def __init__(self, hidden_size, n_classes):
        super().__init__()
        self.fc = nn.Linear(hidden_size, n_classes)
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, embeddings):
        logits = self.fc(embeddings)
        return logits
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, encoder_stride, num_channels):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=encoder_stride**2 * num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(encoder_stride),
        )
        self.activation = nn.Sigmoid()    

    def forward(self, x):
        batch_size, sequence_length, num_channels = x.shape
        height = width = math.floor(sequence_length**0.5)
        x = x.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        x = self.decoder(x)
        x = self.activation(x)
        return x
