#%%
import torch
from configs.config import ConfigSelector
from utils.dataset_utils import DatasetSplitter
from utils.model_utils import GeneratePredictions
from utils.simmim_utils import DataLoaderGenerator, RsiMlcDataset, RsiMccDataset
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
    
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

model_size = 'base'
learning_task = 'mlc'
n_classes = 17
w = 0.5

args = ConfigSelector()

seed_all(args.seed)
dataset = f'{args.dataset}'
path =  f'../rs_datasets/{args.learning_task}/{args.dataset}'

fraction_labeled = 0.05
rs_datasets = {"mcc": RsiMccDataset, "mlc": RsiMlcDataset}
dss = DatasetSplitter(path, args.learning_task, dataset, fraction_labeled=fraction_labeled, seed=args.seed)
dataframes = dss.create_dataframes()

dataloaders = DataLoaderGenerator(
    rs_dataset=rs_datasets[args.learning_task], 
    dataframes=dataframes, 
    batch_size=args.batch_size,
    image_size=args.image_size
    ).get_data_loaders() 

device = get_default_device()

model = ssl_mae_deit(model_size, learning_task, n_classes, w)


model = model.to(device)
predictor = GeneratePredictions()
df = predictor.run_prediction(model, dataloaders['test'], device)

batch = next(iter(dataloaders['test']))
images, masks, labels = batch
outputs = model(images.to(device), labels.to(device), masks.to(device))

IMAGE_COLOR_MEAN = (0.485, 0.456, 0.406)
IMAGE_COLOR_STD = (0.229, 0.224, 0.225)
patch_size = 16

recon = outputs['x_rec'].cpu()
mask  = outputs['mask'].cpu()
mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)
mask = unpatchify(mask, patch_size)

imgs = images / 255.0

t = UnNormalise(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD)
recon = t(recon)

plt.figure(0)
plt.imshow(make_grid_image(recon))

plt.figure(1)
im_masked = imgs * (1 - mask)
plt.imshow(make_grid_image(im_masked))

plt.figure(2)
# MAE reconstruction pasted with visible patches
im_paste = imgs * (1 - mask) + recon * mask

plt.imshow(make_grid_image(im_paste))
