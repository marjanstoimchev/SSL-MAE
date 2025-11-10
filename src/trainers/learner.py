import torch
import lightning as L
from typing import Optional
from src.utils.simmim_utils import DataLoaderGenerator
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader

from torch import Tensor
from typing import Any, List, Optional, Union

class SSLMAE_Learner(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        # Map modes to their corresponding methods
        self.training_strategies = {
            'supervised': self.process_supervised,
            'semi_supervised': self.process_semi_supervised
        }

    def forward(self, x, mask, non_masked = None, y = None):
        return self.model(x, mask, non_masked, y)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        # Call the corresponding method based on the current mode
        x, mask, non_masked, y = self.training_strategies[self.config.mode](batch)
        loss = self(x, mask, non_masked, y)['loss']
        self.log_dict({'loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x.size(0))
        return loss

    def process_supervised(self, batch):
        x, mask, non_masked, y = batch
        return x, mask, non_masked, y

    def process_semi_supervised(self, batch):
        (x, mask_x, non_masked, y), (u, mask_u) = batch
        x = torch.cat((x, u), dim=0)
        mask = torch.cat((mask_x, mask_u), dim=0)
        return x, mask, non_masked, y
        
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, mask, non_masked, y = batch
        loss = self(x, mask, non_masked, y)['loss']
        self.log_dict({'val_loss': loss}, on_step=True, on_epoch=True, sync_dist=True, batch_size=x.size(0))
    
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx=0) -> None:
        x, mask, non_masked, y = batch
        logits = self(x, mask, non_masked, y)['logits']
        return logits

    def configure_optimizers(self):  
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.lr, betas=(0.9, 0.95)
            )

        # Initialize the return configuration with the optimizer.
        optimizer_config = {
            "optimizer": optimizer,
        }        
        # If apply_scheduler is True, add the scheduler configuration.
        if self.config.apply_scheduler:            
            self.trainer.fit_loop.setup_data()                

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=1e-4, 
                        total_steps = self.trainer.estimated_stepping_batches,
                        pct_start = 0.01,
                        )
            
            # Add the scheduler configuration.
            optimizer_config["lr_scheduler"] = {
                "scheduler": scheduler,
                'name': 'train/lr',
                "interval": "step",  # or "epoch" based on your scheduling
                "monitor": "val_loss",
                "frequency": 1  # This aligns with the default 'check_val_every_n_epoch=1'
            }

        return optimizer_config

class SSLDataModule(LightningDataModule):
    def __init__(self, rs_dataset, dataframes, batch_size, image_size, mask_ratio, mode):
        super().__init__()
        self.rs_dataset = rs_dataset
        self.dataframes = dataframes
        self.batch_size = batch_size
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.mode = mode

    def setup(self, stage: Optional[str] = None):
        # Assuming your DataLoaderGenerator handles the creation of DataLoaders directly
        # and doesn't require splitting here.

        dataloaders = DataLoaderGenerator(
                rs_dataset=self.rs_dataset, 
                dataframes=self.dataframes, 
                batch_size=self.batch_size,
                image_size=self.image_size,
                mask_ratio=self.mask_ratio,
                mode = self.mode
                ).get_data_loaders() 
        
        self.dataloaders = dataloaders
                
    def train_dataloader(self):
        return self.dataloaders[self.mode]

    def val_dataloader(self):
        return self.dataloaders['val']

    # Include predict_dataloader if your DataLoaderGenerator also provides for a prediction set
    def predict_dataloader(self):
        return self.dataloaders['test']