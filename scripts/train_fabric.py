#!/usr/bin/env python
"""Training script for SSL-MAE using Lightning Fabric."""

import os
import sys
import torch
import wandb

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.configs.config import ConfigSelector
from src.utils.dataset_utils import DatasetSplitter
from src.utils.simmim_utils import RsiMccDataset, RsiMlcDataset, DataLoaderGenerator
from src.models.ssl_mae import ssl_mae
from src.trainers.fabric_learner import SSLMAE_Learner

import lightning as L
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from src.utils.utils import seed_everything, get_optimizer

args = ConfigSelector()
path =  f'../rs_datasets/{args.learning_task}/{args.dataset}'
dss = DatasetSplitter(path, args.learning_task, args.dataset, fraction_labeled=0.1, seed=args.seed)
dataframes = dss.create_dataframes()
#%%
dataloaders = DataLoaderGenerator(
                rs_dataset=RsiMlcDataset, 
                dataframes=dataframes, 
                batch_size=args.batch_size,
                image_size=args.image_size,
                mask_ratio=args.mask_ratio,
                mode = 'semi_supervised'
                ).get_data_loaders()

from tqdm import tqdm

for key, dl in dataloaders.items():
    print(key, len(dl))

#%%
x = next(iter(dataloaders['semi_supervised']))
#%%

ds = RsiMlcDataset(
    dataframes['labeled'],
    #dataframes['unlabeled'],
    data_type = 'both',
    input_size=224, 
    mask_patch_size=16, 
    model_patch_size=16, 
    mask_ratio=0.75
    )
print(len(ds))


#%%
for batch in dataloaders['labeled']:
    print(batch[3].shape)
#%%

import numpy as np
def select_batch(data, idx):
    return data[idx]

def convert_batch(batch):
    x = torch.stack(list(map(lambda x: select_batch(x, 0), batch[0])), 0)
    mask = torch.stack(list(map(lambda x: select_batch(x, 1), batch[0])), 0)
    non_masked = torch.stack(list(map(lambda x: select_batch(x, 2), batch[0])), 0)
    y = torch.tensor(np.array(batch[1]))
    return x, mask, non_masked, y

x, mask, non_masked, y = convert_batch(batch)
print(x.shape)
print(mask.shape)
print(non_masked.shape)
print(y.shape)
#%%
for epoch in range(1):
    p_bar = tqdm(
                enumerate(zip(dataloaders['labeled'], dataloaders['unlabeled'])),
                total=len(dataloaders['unlabeled']),
                desc=f"Epoch {epoch + 1} - Train Step: ", position=0, leave=True
                    )

    for step, batches in p_bar:
        batch_x, batch_u = batches
        #%%
        #%%
        print(batch_x[0].shape, batch_u[0].shape)
#%%

model = ssl_mae(
        architecture=args.architecture,
        model_size=args.model_size, 
        learning_task=args.learning_task, 
        n_classes=args.n_classes, 
        w=0.5)

#%%
model(*batch_x)

#%%
def main(args, fraction_labeled, w = 0.5):

    seed_everything(args.seed)
    #ddp = DDPStrategy(process_group_backend='gloo', find_unused_parameters=False)

    fabric = L.Fabric(
        devices=[0, 1, 3, 7], 
        accelerator = 'gpu', 
        strategy = 'ddp', 
        precision = '16-mixed',
        )
    

    path =  f'../rs_datasets/{args.learning_task}/{args.dataset}'
    dss = DatasetSplitter(path, args.learning_task, args.dataset, fraction_labeled=fraction_labeled, seed=args.seed)
    dataframes = dss.create_dataframes()

    dataloaders = DataLoaderGenerator(
                rs_dataset=RsiMlcDataset, 
                dataframes=dataframes, 
                batch_size=args.batch_size,
                image_size=args.image_size,
                mask_ratio=args.mask_ratio,
                ).get_data_loaders() 
    
    fabric.launch()
    fabric.barrier()
    
    model = ssl_mae(
        architecture=args.architecture,
        model_size=args.model_size, 
        learning_task=args.learning_task, 
        n_classes=args.n_classes, 
        w=w)
    
    optimizer = get_optimizer(model, args.lr)
    
    model, optimizer = fabric.setup(model, optimizer)
    labeled_dl, unlabeled_dl, val_dl, test_dl = fabric.setup_dataloaders(
        dataloaders['labeled'], 
        dataloaders['unlabeled'], 
        dataloaders['val'], 
        dataloaders['test'], 
        #use_distributed_sampler=False
        )

    dataloaders = {
        "labeled": labeled_dl,
        "unlabeled": unlabeled_dl,
        "val": val_dl,
        "test": test_dl
        }
    
    fabric.seed_everything(args.seed)

    learner = SSLMAE_Learner(
        args.epochs,
        model, 
        optimizer,
        fabric,
        lr=args.lr,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        n_accumulate=args.n_accumulate,
        min_lr=args.min_lr,
        mode=args.mode,
        save_model_path=f"saved_models_fabric/{args.mode}",
    )
    
    learner.fit(dataloaders, logger = True)

if __name__ == '__main__':
    args = ConfigSelector()
    main(args, fraction_labeled=0.1, w = None)