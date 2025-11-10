import torch
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data._utils.collate import default_collate
from utils.samplers import RandomSampler, DistributedProxySampler, BatchSampler
#from torch.utils.data.sampler import BatchSampler, RandomSampler

class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".
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

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        # Create an array of all indices
        all_indices = np.arange(self.token_count)
        # Find indices that are not in 'mask_idx' using a boolean mask
        non_mask_idx = np.isin(all_indices, mask_idx, invert=True)
        # Extract the non-masked indices
        non_mask_idx = all_indices[non_mask_idx]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten()), torch.tensor(non_mask_idx)
    
class SimMIMTransform:
    def __init__(self, data_type, **kwargs):
        if data_type in ['labeled', 'both']:
            # Apply data augmentation for training data
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.RandomResizedCrop(kwargs['input_size'], scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.Resize((kwargs['input_size'], kwargs['input_size'])),
                T.ToTensor(),
            ])
        elif data_type in ['val', 'test']:
            # No data augmentation for validation and test data
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((kwargs['input_size'], kwargs['input_size'])),
                T.ToTensor(),
            ])
        
        self.mask_generator = MaskGenerator(**kwargs)

    def __call__(self, img):
        img = self.transform_img(img)
        mask, non_masked = self.mask_generator() if self.mask_generator else None
        return img, mask, non_masked

# def collate_fn(batch):
#     if not isinstance(batch[0][0], tuple):
#         return default_collate(batch)
#     else:
#         batch_num = len(batch)
#         ret = []
#         for item_idx in range(len(batch[0][0])):
#             if batch[0][0][item_idx] is None:
#                 ret.append(None)
#             else:
#                 ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
#         ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
#         return ret
    
def collate_fn(instances):
    batch = []

    for i in range(len(instances[0])):
        batch.append([instance[i] for instance in instances])

    return batch

class RsiMlcDataset(torch.utils.data.Dataset):
    def __init__(self, df_labeled, df_unlabeled=None, data_type='both', **kwargs):
        super(RsiMlcDataset, self).__init__()

        # Validate the input parameters and setup the dataset configuration
        if df_unlabeled is None:
            # If no unlabeled DataFrame is provided, ensure we are not using 'both'
            self.df_labeled = df_labeled
            self.data_type = 'labeled'  # Force to 'labeled' if only one DataFrame is provided
            transform_mode = 'labeled'  # Use labeled mode unless specified to be validation or testing
            if data_type in ['val', 'test']:
                transform_mode = 'val'
        else:
            # If both labeled and unlabeled data are provided
            self.df_labeled = df_labeled
            self.df_unlabeled = df_unlabeled
            self.data_type = data_type  # Use the provided data_type
            if data_type == 'both':
                num_rows_large = len(df_unlabeled)
                self.df_labeled = df_labeled.sample(n=num_rows_large, replace=True, random_state=42)
                transform_mode = 'both'
            elif data_type in ['val', 'test']:
                transform_mode = 'val'
            else:
                transform_mode = data_type

            self.data_u = [(row.iloc[0], row.iloc[1:]) for _, row in self.df_unlabeled.iterrows()]

        self.data_x = [(row.iloc[0], row.iloc[1:]) for _, row in self.df_labeled.iterrows()]

        # Initialize transformations based on the decided mode
        self.transform_x = SimMIMTransform(transform_mode, **kwargs)
        if self.data_type == 'both' and df_unlabeled is not None:
            self.transform_u = SimMIMTransform('both', **kwargs)

    def __len__(self):
        if self.data_type == 'both' and hasattr(self, 'df_unlabeled'):
            return len(self.df_unlabeled)
        return len(self.df_labeled)

    def __getitem__(self, index):
        image_path_x, label = self.data_x[index]
        image_x = Image.open(image_path_x).convert('RGB')  # Open the image
        label = np.array(list(label.values))
        image_x, mask_x, non_masked = self.transform_x(image_x)  # Apply the transformation and get the mask

        if self.data_type == 'both' and hasattr(self, 'data_u'):
            image_path_u, _ = self.data_u[index]
            image_u = Image.open(image_path_u).convert('RGB')  # Open the image
            image_u, mask_u, _ = self.transform_u(image_u)
            return (image_x, mask_x, non_masked, torch.tensor(label, dtype=torch.float)), (image_u, mask_u)
        
        return (image_x, mask_x, non_masked, torch.tensor(label, dtype=torch.float))



class RsiMccDataset(torch.utils.data.Dataset):

    def __init__ (self, df, data_type = 'labeled', **kwargs):
        super(RsiMccDataset, self).__init__()
        self.df = df
        self.data = [(row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4:].tolist()) for _, row in self.df.iterrows()]
        self.transform = SimMIMTransform(data_type, **kwargs)  # Instantiate the transformation with the configuration

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, index):
        image_path, label = self.data[index]
        label = np.array(list(label.values))  
        image = Image.open(image_path) # Open the image
        image, mask = self.transform(image)  # Apply the transformation and get the mask
        return (image, mask), label  # Return both image, mask, and label

class DataLoaderGenerator:
    """
    Generates DataLoader instances for various subsets of a dataset with appropriate transformations.
    """
    def __init__(self, rs_dataset, dataframes, batch_size, image_size, mask_ratio=0.75, mode='supervised'):
        self.rs_dataset = rs_dataset
        self.dataframes = dataframes
        self.batch_size = batch_size
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.mode = mode

        # Control behavior based on the mode and type of data
        self.shuffle = {'train': True, 'val': False, 'test': False}
        self.drop_last = {'train': True, 'val': False, 'test': False}
        
        self.print_dataframe_stats()

    def print_dataframe_stats(self):
        for split, df in self.dataframes.items():
            print(f"The length of the {split} dataframe is {len(df)}")

    def get_data_loaders(self):
        """
        Creates DataLoaders for all specified data types and modes.

        Returns:
            dict: A dictionary of DataLoaders for each data type.
        """
        loaders = {}

        if self.mode == 'semi_supervised':
            # In semi_supervised mode, both labeled and unlabeled data are necessary
            dataset_both = self.rs_dataset(
                self.dataframes['labeled'],
                self.dataframes['unlabeled'],
                data_type='both',
                input_size=self.image_size,
                mask_patch_size=16,
                model_patch_size=16,
                mask_ratio=self.mask_ratio
            )
            loaders['semi_supervised'] = DataLoader(
                dataset_both,
                batch_size=self.batch_size,
                shuffle=self.shuffle['train'],
                drop_last=self.drop_last['train'],
                pin_memory=True,
                num_workers=16
            )

        if self.mode == 'supervised':
            # In supervised mode, only labeled data is used
            dataset_labeled = self.rs_dataset(
                self.dataframes['labeled'],
                data_type='labeled',
                input_size=self.image_size,
                mask_patch_size=16,
                model_patch_size=16,
                mask_ratio=self.mask_ratio
            )
            loaders['supervised'] = DataLoader(
                dataset_labeled,
                batch_size=self.batch_size,
                shuffle=self.shuffle['train'],
                drop_last=self.drop_last['train'],
                pin_memory=True,
                num_workers=16
            )

        # Create DataLoaders for validation and test data with no masking ratio
        for data_type in ['val', 'test']:
            if data_type in self.dataframes:
                dataset = self.rs_dataset(
                    self.dataframes[data_type],
                    data_type=data_type,
                    input_size=self.image_size,
                    mask_patch_size=16,
                    model_patch_size=16,
                    mask_ratio=0.0  # No masking for validation and testing
                )
                loaders[data_type] = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle[data_type],
                    drop_last=self.drop_last[data_type],
                    pin_memory=True,
                    num_workers=16
                )

        return loaders