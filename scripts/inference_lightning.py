#%%
import os
import glob
import torch
from configs.config import ConfigSelector
from utils.dataset_utils import DatasetSplitter
from utils.model_utils import GeneratePredictions
from utils.simmim_utils import DataLoaderGenerator, RsiMlcDataset, RsiMccDataset
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from lightning import seed_everything
from model.ssl_mae import ssl_mae
from model.learner import SSLMAE_Learner, SSLDataModule
import lightning as L

def make_grid_image(image):
    grid_img = make_grid(image).permute(1,2,0).detach().numpy()
    return grid_img

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

args = ConfigSelector()

seed_everything(args.seed, workers=True)
dataset = f'{args.dataset}'
path =  f'../rs_datasets/{args.learning_task}/{args.dataset}'

fraction_labeled = 0.1
rs_datasets = {"mcc": RsiMccDataset, "mlc": RsiMlcDataset}
dss = DatasetSplitter(path, args.learning_task, dataset, fraction_labeled=fraction_labeled, seed=args.seed)
dataframes = dss.create_dataframes()

dataloaders = DataLoaderGenerator(
    rs_dataset=rs_datasets[args.learning_task], 
    dataframes=dataframes, 
    batch_size=args.batch_size,
    image_size=args.image_size
    ).get_data_loaders() 

trainer = L.Trainer(
                enable_model_summary=True,
                num_sanity_val_steps=0,
                accelerator='auto', 
                devices=[3],
                precision="16-mixed",
            )

model = ssl_mae(
        architecture=args.architecture,
        model_size=args.model_size, 
        learning_task=args.learning_task, 
        n_classes=args.n_classes, 
        w=None)
    
dirpath = 'saved_models/semi_supervised'
print(f"Loading model from path: {dirpath}")
cofig = ConfigSelector()
PATH = glob.glob(os.path.join(dirpath, '**/*.ckpt'), recursive=True)[0]
lightning_model = SSLMAE_Learner.load_from_checkpoint(
    PATH, model = model, config = cofig
    )

from tqdm import tqdm
from utils.utils import calculate_mlc_metrics, calculate_mcc_metrics

dataloader = dataloaders['test']
pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Getting the targets ---: ", position=0, leave=True)
        
probas, targets = [], []

targets = []
for _, data in pbar:
    _, _, _, y = data
    targets.append(y)

targets = torch.cat(targets, dim=0).cpu().detach().float()
logits = trainer.predict(lightning_model, dataloaders['test'])
logits = torch.cat(logits, dim=0).cpu().detach()
probas = torch.sigmoid(logits).data
preds = (probas.numpy() > 0.5).astype('int')

Y = dict()
Y['y_scores'] = probas
Y['y_pred']   = preds
Y['y_true']   = targets

df = calculate_mlc_metrics(Y)
print(df)