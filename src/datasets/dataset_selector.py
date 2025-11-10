import pandas as pd
from prettytable import PrettyTable
from datasets.ankara_dataset import AnkaraDataset
from datasets.ucm_dataset import UCMDataset
from datasets.aid_dataset import AIDDataset
from datasets.dfc_15_dataset import DFC15Dataset
from datasets.mlrsnet_dataset import MLRSNetDataset
from datasets.ben_dataset import BEN_Dataset_Full_43, BEN_Dataset_Full_19

class DatasetSelectorMLC:
    def __init__(self, path, dataset_name):
        self.path = path
        self.dataset_name = dataset_name
        self.dataset_classes = {
            'Ankara': AnkaraDataset,
            'UCM_mlc': UCMDataset,
            'AID_mlc': AIDDataset,
            'DFC_15': DFC15Dataset,
            'MLRSNet': MLRSNetDataset,
            'BEN_43': BEN_Dataset_Full_43,
            'BEN_19': BEN_Dataset_Full_19
        }

    def select(self):
        """
        Selects and initializes the dataset class based on the provided dataset name.
        
        Returns:
            An instance of the selected dataset class.
        """
        dataset_class = self.dataset_classes.get(self.dataset_name)
        if not dataset_class:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
        return dataset_class(self.path)

    def generate(self):
        """
        Generates and prints the dataset information using PrettyTable and loads the dataframe(s)
        based on the dataset selection.
        
        Returns:
            tuple: Depending on the dataset, returns a single dataframe or a tuple of train and test dataframes.
        """
        ds = self.select()

        if self.dataset_name in ['DFC_15', 'AID_mlc']:
            df, df_test = ds.load_dataframe()
            table = self._prepare_table([self.dataset_name, len(df), len(df_test)], ["Dataset", "Train", "Test"])
        else:
            df = ds.load_dataframe()
            table = self._prepare_table([self.dataset_name, len(df)], ["Dataset", "Train"])
            df_test = None

        print(table)
        return df, df_test

    def _prepare_table(self, row_data, field_names):
        """
        Prepares a PrettyTable instance with provided data.
        
        Parameters:
            row_data (list): Data to be added as a row in the table.
            field_names (list): List of field names for the table headers.
        
        Returns:
            PrettyTable: A PrettyTable object with the added row data.
        """
        table = PrettyTable()
        table.field_names = field_names
        table.add_row(row_data)
        return table