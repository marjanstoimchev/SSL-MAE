import os
import math
import numpy as np
import pandas as pd
from typing import Tuple
from datasets.dataset_selector import DatasetSelectorMLC
from datasets.generate import GenerateCSV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from utils.utils import create_path

def create_label_encoder(df: pd.DataFrame, encoder_classes_path: str) -> pd.DataFrame:
    """
    Encodes the 'label' column of a DataFrame to numeric IDs, saves the encoder's classes
    at the specified path, and returns the updated DataFrame with an additional 'label_id' column.

    Parameters:
    - df (DataFrame): DataFrame containing a 'label' column.
    - encoder_classes_path (str): File system path where the label encoder's classes
      are saved as a NumPy .npy file.

    Returns:
    - DataFrame: The original DataFrame with an added 'label_id' column representing
      encoded labels.

    Raises:
    - ValueError: If the 'label' column is missing from the DataFrame.
    """
    if 'label' not in df.columns:
        raise ValueError("'label' column missing in DataFrame.")
    
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df["label"])
    os.makedirs(os.path.dirname(encoder_classes_path), exist_ok=True)
    np.save(encoder_classes_path, encoder.classes_)
    
    return df

def to_one_hot(df: pd.DataFrame, enc: OneHotEncoder) -> pd.DataFrame:
    """
    One-hot encodes 'label_id' in df, concatenates it with original df.
    
    Parameters:
    - df (DataFrame): DataFrame with 'label_id'.
    - enc (OneHotEncoder): Unfitted OneHotEncoder instance.
    
    Returns:
    - DataFrame: Enhanced DataFrame with one-hot encoded columns.
    """
    if 'label_id' not in df:
        raise ValueError("'label_id' column required.")
    encoded = enc.fit_transform(df[['label_id']]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out())
    return pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

def one_hot_encode(labels, num_classes):
    """
    One-hot encodes numeric labels into a matrix.
    
    Parameters:
    - labels (array-like): Labels between 0 and num_classes-1.
    - num_classes (int): Number of classes.
    
    Returns:
    - numpy.ndarray: One-hot encoded matrix.
    """
    labels = np.asarray(labels)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Labels out of range.")
    return np.eye(num_classes)[labels]



def split_mlc_df(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple:
    """
    Splits a DataFrame for multi-label classification (MLC) tasks into training and test sets.
    
    Maintains the distribution of labels across the splits using MultilabelStratifiedShuffleSplit.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to split. Assumes the first column is features and the rest are labels.
    - test_size (float): Fraction of the dataset to include in the test split.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - tuple of pd.DataFrame: (training DataFrame, testing DataFrame).
    """
    # Reset index to ensure consistent slicing
    df.reset_index(drop=True, inplace=True)
    # Features are all columns except the last, labels are assumed to be in the last column
    X = df.iloc[:, 0]
    y = df.iloc[:, 1:-1].values
    # Initialize and apply the split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(msss.split(X, y))
    # Return the split DataFrames
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)



def split_mcc_df(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple:
    """
    Splits a DataFrame for multi-class classification (MCC) into training and test sets,
    maintaining the distribution of the 'label_id' column.
    
    Assumes the DataFrame has a 'label_id' column for stratification.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to split. Must contain a 'label_id' column.
    - test_size (float): Fraction of the dataset to include in the test split.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - tuple of pd.DataFrame: (training DataFrame, testing DataFrame).
    """
    # Ensure DataFrame's index is reset for consistent data access
    df.reset_index(drop=True, inplace=True)
    # Initialize StratifiedShuffleSplit instance
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # Perform stratified split based on 'label_id'
    train_idx, test_idx = next(sss.split(df, df['label_id']))
    # Return the training and testing set DataFrames
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def random_sampling(df: pd.DataFrame, p: float, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly splits a DataFrame into labeled and unlabeled sets.

    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        p (float): The proportion of the DataFrame to remain unlabeled.
        seed (Optional[int]): Seed for random state to ensure reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the labeled and unlabeled DataFrames.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if p == 1:
        return df, pd.DataFrame(columns=df.columns)  # Return empty DataFrame for unlabeled

    p_labeled = p
    n_labeled = math.ceil(len(df) * p_labeled)

    # Ensure at least one sample in each set if possible
    n_labeled = max(min(n_labeled, len(df) - 1), 1)
    
    indices = np.random.permutation(df.index)
    indices_labeled = indices[:n_labeled]
    indices_unlabeled = indices[n_labeled:]
    
    labeled = df.loc[indices_labeled]
    unlabeled = df.loc[indices_unlabeled]

    return labeled, unlabeled



def prepare_dataframes_mlc(path, dataset):
    """
    Prepares the dataframes for a multi-label classification (MLC) task based on the specified dataset.
    
    Parameters:
    - path (str): The base path where the dataset is located.
    - dataset (str): The name of the dataset to be used.
    
    Returns:
    - tuple: A tuple containing the training dataframe and testing dataframe.
    """
    data_select = DatasetSelectorMLC(path, dataset)
    df_train, df_test = data_select.generate()
    return df_train, df_test



def prepare_dataframes_mcc(path, dataset):
    """
    Prepares the dataframes for a multi-class classification (MCC) task based on the specified dataset and path.
    
    Parameters:
    - path (str): The base path where the dataset is located.
    - dataset (str): The name of the dataset to be prepared.
    
    Returns:
    - tuple: A tuple containing the training dataframe and testing dataframe (if applicable).
    """
    path_labels = create_path(f"encoder_classes/{dataset}")
    df_train, df_test = None, None

    if dataset == "AID_mcc":
        # Specific handling for the "AID_mcc" dataset
        path_tr = f"{path}/images/images_tr"
        path_te = f"{path}/images/images_test"
        df_train = GenerateCSV(path_tr).generate()
        df_test = GenerateCSV(path_te).generate()       
        df_test['file_path'] = df_test['file_path'] + '/' + df_test['image_name']
    
    else:
        # General handling for other datasets
        df_train = GenerateCSV(path).generate()
    # Common processing for df_train across all datasets
    df_train['file_path'] = df_train['file_path'] + '/' + df_train['image_name']
    
    # Apply label encoding to both train and test dataframes if df_test exists
    df_train = create_label_encoder(df_train, f"{path_labels}/classes.npy")
    if df_test is not None:
        df_test = create_label_encoder(df_test, f"{path_labels}/classes.npy")

    
    return df_train, df_test



class DatasetSplitter:
    """Utility class for dataset manipulation."""
    
    def __init__(self, path, learning_task, dataset, fraction_labeled, seed):
        """Initialize the DatasetUtils instance with a fraction for sampling and a random seed."""
        super().__init__()
        self.path = path
        self.learning_task = learning_task
        self.dataset = dataset
        self.fraction_labeled = fraction_labeled
        self.seed = seed

    def _split_and_sample(self, df_train, df_test, learning_task):
        """Splits the training data and samples it according to the task type (mlc or mcc)."""
        if df_test is None:
            df_train, df_test = self._split_df(df_train, learning_task=learning_task)

        df_train, df_val = self._split_df(df_train, learning_task=learning_task, test_size=0.1)

        if learning_task == "mlc" and self.dataset == "AID_mlc":
            df_val = self._ensure_label_presence(df_train, df_val, 'mobile-home')

        df_labeled, df_unlabeled = random_sampling(df_train, p=self.fraction_labeled, seed=self.seed)

        return {
            'labeled': df_labeled,
            'unlabeled': df_unlabeled,
            'val': df_val,
            'test': df_test
        }

    def _split_df(self, df, learning_task, test_size=0.2):
        """Splits a dataframe into training and testing sets based on the task type."""
        if learning_task == "mlc":
            return split_mlc_df(df, test_size=test_size, seed=self.seed)
        elif learning_task == "mcc":
            return split_mcc_df(df, test_size=test_size, seed=self.seed)
        
    def sample_mlc(self, df_train, df_test=None):
        """Samples the multi-label classification (MLC) dataset."""
        return self._split_and_sample(df_train, df_test, learning_task='mlc')

    def sample_mcc(self, df_train, df_test=None):
        """Samples the multi-class classification (MCC) dataset, applying one-hot encoding."""
        enc = OneHotEncoder()
        df_train = to_one_hot(df_train, enc)  
        if df_test is not None:  
            df_test = to_one_hot(df_test, enc)
        return self._split_and_sample(df_train, df_test, learning_task="mcc")
    
    def create_dataframes(self):
        """Creates and samples dataframes based on the learning task."""
        
        if self.learning_task == "mlc":
            df_train, df_test = prepare_dataframes_mlc(self.path, self.dataset)
        elif self.learning_task == "mcc":
            df_train, df_test = prepare_dataframes_mcc(self.path, self.dataset)

        if self.learning_task in ['mlc', 'mcc']:
            return getattr(self, f'sample_{self.learning_task}')(df_train, df_test)
        else:
            raise ValueError(f"Unsupported learning task: {self.learning_task}")
        
    def _ensure_label_presence(self, df_train, df_val, label):
        """Ensures that the validation set contains at least one instance of a specific label."""
        if df_val.iloc[:, 1:].sum(0)[label] == 0:
            new_row = df_train[df_train[label] == 1]
            df_val = pd.concat([df_val, new_row], ignore_index=True)
            df_val.reset_index(inplace = False)
        return df_val
