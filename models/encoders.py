import torch
from transformers import (
    DeiTModel, ViTModel, 
    DeiTPreTrainedModel, 
    ViTPreTrainedModel, 
    )

class DeiTEncoder(DeiTPreTrainedModel):
    def __init__(self, config, global_avg_pool = True):
        super().__init__(config)

        #deit = DeiTForImageClassification(config).deit
        deit = DeiTModel(config, add_pooling_layer=False, use_mask_token=True)
        self.deit = deit
        self.hidden_size = deit.config.hidden_size
        self.global_avg_pool = global_avg_pool
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, bool_masked_pos, ids_keep = None):
        outputs = self.deit(x, bool_masked_pos = bool_masked_pos)
        sequence_output = outputs.last_hidden_state[:, 1:-1]

        sequence_output_visible = torch.gather(
            sequence_output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.hidden_size)
            )

        if self.global_avg_pool:
            embeddings = sequence_output_visible.mean(1)
        else:
            embeddings = outputs.last_hidden_state[:, 0]
        return sequence_output, embeddings
    
class ViTEncoder(ViTPreTrainedModel):
    def __init__(self, config, global_avg_pool = True):
        super().__init__(config)

        #deit = DeiTForImageClassification(config).deit
        vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
        self.vit = vit
        self.hidden_size = vit.config.hidden_size
        self.global_avg_pool = global_avg_pool
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, bool_masked_pos, ids_keep = None):
        outputs = self.vit(x, bool_masked_pos = bool_masked_pos)
        sequence_output = outputs.last_hidden_state[:, 1:]

        sequence_output_visible = torch.gather(
            sequence_output, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.hidden_size)
            )

        if self.global_avg_pool:
            embeddings = sequence_output_visible.mean(1)
        else:
            embeddings = outputs.last_hidden_state[:, 0]
        return sequence_output, embeddings
