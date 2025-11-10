#!/usr/bin/env python
"""Training script for SSL-MAE using PyTorch Lightning."""

import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.configs.config import ConfigSelector
from src.utils.dataset_utils import DatasetSplitter
from src.utils.simmim_utils import RsiMccDataset, RsiMlcDataset
from src.models.ssl_mae import ssl_mae
from src.trainers.learner import SSLMAE_Learner, SSLDataModule
from src.trainers.callbacks import ModelCheckpoint_, EarlyStopping_, RichProgressBar_

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning import seed_everything



#%%
def get_trainner(args, strategy, devices, wandb_logger, callbacks):
    trainer = L.Trainer(
                enable_model_summary=True,
                num_sanity_val_steps=0,
                accelerator='auto', 
                strategy=strategy,
                devices=devices,
                precision="16-mixed",
                min_epochs = 5,
                max_epochs = args.epochs,
                accumulate_grad_batches = args.n_accumulate,
                sync_batchnorm = True,\
                benchmark = True,
                log_every_n_steps = 1,
                enable_checkpointing=True,
                use_distributed_sampler=False, # This is a must if using BatchSampler
                logger=wandb_logger,
                callbacks=callbacks
            )
    return trainer

def main(args, fraction_labeled, w = 0.5):
    seed_everything(args.seed, workers=True)
    path =  f'../rs_datasets/{args.learning_task}/{args.dataset}'
    dss = DatasetSplitter(path, args.learning_task, args.dataset, fraction_labeled=fraction_labeled, seed=args.seed)
    dataframes = dss.create_dataframes()

    ssl_dm = SSLDataModule(
        rs_dataset=RsiMlcDataset, 
        dataframes=dataframes,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mask_ratio=args.mask_ratio,
        mode=args.mode
        )

    ddp = DDPStrategy(
        process_group_backend='gloo', find_unused_parameters=False,
        )

    wandb_logger = WandbLogger(project='SSL-MAE-lightning', name=f"{args.mode}")
    dirpath = os.path.join(*[args.save_model_dir, args.mode])
                            
    callbacks = [
            LearningRateMonitor(logging_interval='step'),
            RichProgressBar_(),
            EarlyStopping_(metric="val_loss", mode = "min", patience = args.patience),
            ModelCheckpoint_(
                dirpath=dirpath, 
                metric="val_loss", 
                mode = "min",
                save_on_train_epoch_end = False),
            ]
    
    model = ssl_mae(
        architecture=args.architecture,
        model_size=args.model_size, 
        learning_task=args.learning_task, 
        n_classes=args.n_classes, 
        w=w)
    
    lightning_model = SSLMAE_Learner(model, args)
    trainer = get_trainner(args, strategy=ddp, devices = [0, 1, 3, 7], wandb_logger=wandb_logger, callbacks=callbacks)
    trainer.fit(lightning_model, datamodule = ssl_dm)

if __name__ == '__main__':
    args = ConfigSelector()
    main(args, fraction_labeled=0.1, w = None)