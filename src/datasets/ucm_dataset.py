import os, glob
import numpy as np
import pandas as pd
from utils.utils import SortedAlphanumeric
from configs.config import BaseConfig
import imghdr

class UCMDataset:
    def __init__(self, base_dir, extension="tif"):  # Set a default image extension
        super().__init__()
        self.images_path = base_dir#os.path.join(base_dir, 'mlc', 'UCM')
        self.labels_path = os.path.join(self.images_path, 'LandUse_Multilabeled.txt')
        self.extension = extension  # Specify the image file extension
    
    def load_dataframe(self):
        df = pd.read_csv(self.labels_path, delimiter = "\t")
        directories = [x[0] for x in os.walk(self.images_path)][1:]
        files = [file for d in directories for file in glob.glob(os.path.join(d, '*' + self.extension))]
        files = SortedAlphanumeric(files).sort()
        df['IMAGE\LABEL'] = files   
        return df

