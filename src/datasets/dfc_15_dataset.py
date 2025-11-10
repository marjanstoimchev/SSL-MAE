import glob, os, re
import pandas as pd
from utils.utils import SortedAlphanumeric
from configs.config import BaseConfig

class DFC15Dataset(BaseConfig):
    def __init__(self, extension = "png"):
        super().__init__()
        self.images_path = self.base_dir + '/mlc/DFC_15'
        self.labels_path = self.images_path + '/' + 'multilabel.txt'
        self.extension = extension
        
    def load_dataframe(self):
        
        df = pd.read_csv(self.labels_path, delimiter = "\t")
        directories = [x[0] for x in os.walk(self.images_path)][1:]
        test_images, train_images = directories # test_images, train_images
        
        files_train = glob.glob(os.path.join(train_images, '*' + self.extension))
        files_test = glob.glob(os.path.join(test_images, '*' + self.extension))
        
        files_train = SortedAlphanumeric(files_train).sort()
        files_test  = SortedAlphanumeric(files_test).sort()
        
        targets_train = [int(re.split(r'(\d+)', file.split('\\')[-1])[1]) - 1 for file in files_train]
        targets_test = [int(re.split(r'(\d+)', file.split('\\')[-1])[1]) - 1 for file in files_test]

        df_train = df.iloc[targets_train].reset_index(drop=True)
        df_test = df.iloc[targets_test].reset_index(drop=True)
        
        df_train['IMAGE\LABEL'] = files_train
        df_test['IMAGE\LABEL'] = files_test

        return df_train, df_test
   
