import glob, os, re
import pandas as pd
import numpy as np
from utils.utils import SortedAlphanumeric, flatten

class GenerateCSV(object):
    def __init__(self, path):
        self.path = path

    def generate(self):
        label_names, image_names, file_paths = [], [], []   
        for dirpath, dirnames, filenames in os.walk(self.path):
            if not dirnames:
                label_name = re.split(r'[\\/]', dirpath)[-1]
                file_paths += [list(np.repeat(dirpath, len(filenames)))]
                label_names += [list(np.repeat(label_name, len(filenames)))]
                image_names += [SortedAlphanumeric(filenames).sort()]
                
            
        image_names = list(flatten(image_names))
        label_names = list(flatten(label_names))
        file_paths  = list(flatten(file_paths))
        df = pd.DataFrame({"file_path": file_paths, "image_name": image_names, "label": label_names})
        return df
