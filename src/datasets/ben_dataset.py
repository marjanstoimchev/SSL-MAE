import glob, os, re
import pandas as pd
import numpy as np
    
class BEN_Dataset_Full_43:
    def __init__(self):
        self.path_df = f"../ben_dataset_43.txt"

    def load_dataframe(self):
        df = pd.read_csv(self.path_df, sep = ",", header = None)
        df = df.drop(columns = [df.shape[1] - 1])
        return df

class BEN_Dataset_Full_19:
    def __init__(self):
        self.path_df = f"../ben_dataset_19.txt"

    def load_dataframe(self):
        df = pd.read_csv(self.path_df, sep = ",", header = None)
        df = df.drop(columns = [df.shape[1] - 1])
        return df
