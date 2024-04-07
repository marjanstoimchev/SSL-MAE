#%%
import random
import glob, re, os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score,f1_score

class SortedAlphanumeric(object):
    def __init__(self, data):
        super(SortedAlphanumeric, self).__init__()
        self.data = data
        
    def sort(self):  
       """https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir"""
        
       convert = lambda text: int(text) if text.isdigit() else text.lower()
       alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
       return sorted(self.data, key=alphanum_key)
    
class InstantMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path) 

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def seed_all(seed: int = 42):
    print("Using Seed Number {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)  # for numpy pseudo-random generator
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False # False


def extract_number(string):
    # Extracts the integer number from the string using regular expressions
    return int(re.search(r'\d+', string).group())

def alphanumeric_sort(lst):
    # Sorts the list alphanumerically based on the extracted numbers
    return sorted(lst, key=extract_number)

def read_path(path, ext):
    return glob.glob(f"{path}/*.{ext}")

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def findmax(outputs):
    Max = -float("inf")
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index

def OneError(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    num = 0
    one_error = 0
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            Max, index = findmax(outputs[i])
            num = num + 1
            if test_target[i][index] != 1:
                one_error = one_error + 1
    return one_error / num


def calculate_mcc_metrics(Y):
    """
    Calculates various evaluation metrics for multi-class classification.
    
    Parameters:
        Y (dict): A dictionary containing 'y_true' (true labels), 'one_hot' (one-hot encoded true labels),
                  'y_pred' (predicted labels), and 'y_scores' (predicted scores or probabilities).
    
    Returns:
        pd.DataFrame: A DataFrame with metric names as rows and their corresponding values.
    """
    y_true, y_targs_hot, y_pred, y_scores = Y['y_true'], Y['one_hot'], Y['y_pred'], Y['y_scores']
    
    # Define metric names
    metric_names = [
        "micro f1", "micro recall", "micro precision",
        "macro f1", "macro recall", "macro precision",
        "accuracy", "hamming loss", "auprc"
    ]
    
    # Calculate metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_precision = precision_score(y_true, y_pred, average='micro')
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    
    hl = hamming_loss(y_true, y_pred) 
    accuracy = accuracy_score(y_true, y_pred)
    auprc = average_precision_score(y_targs_hot, y_scores, average="macro")
    
    # Map metrics to their names
    metrics = [
        micro_f1, micro_recall, micro_precision,
        macro_f1, macro_recall, macro_precision,
        accuracy, hl, auprc
    ]
    
    dict_metrics = {name: metric for name, metric in zip(metric_names, metrics)}
    df = pd.DataFrame(dict_metrics).T 
    
    return df

def calculate_mlc_metrics(Y):
    """
    Calculates various evaluation metrics for multi-label classification.
    
    Parameters:
        Y (dict): A dictionary containing 'y_true' (true labels), 'y_pred' (predicted labels),
                  and 'y_scores' (predicted scores or probabilities).
    
    Returns:
        pd.DataFrame: A DataFrame with metric names as rows and their corresponding values.
    """
    y_true, y_pred, y_scores = Y['y_true'], Y['y_pred'], Y['y_scores']

    metric_names = ["ranking loss", "one error", "coverage",
                    "average auprc", "weighted auprc",
                    "micro f1", "micro recall", "micro precision",
                    "macro f1", "macro recall", "macro precision",
                    "subset accuracy", "hamming loss",
                    "ml_f_one", "ml_recall", "ml_precision"]

    r_loss = label_ranking_loss(y_true, y_scores)
    oe = OneError(y_scores, y_true)  # Placeholder for an implementation of OneError
    coverage_error_val = coverage_error(y_true, y_scores) - 1  # Adjusting for zero-based index

    average_auprc = average_precision_score(y_true, y_scores, average="macro")
    weighted_auprc = average_precision_score(y_true, y_scores, average="weighted")

    micro_f1 = f1_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average="micro")
    micro_precision = precision_score(y_true, y_pred, average="micro")

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)

    subset_accuracy = accuracy_score(y_true, y_pred)
    hl = hamming_loss(y_true, y_pred)

    # Note: 'samples' averaging may not be appropriate for all metrics if not multi-label classification
    ml_f_one = f1_score(y_true, y_pred, average='samples')
    ml_recall = recall_score(y_true, y_pred, average="samples")
    ml_precision = precision_score(y_true, y_pred, average="samples", zero_division=0)

    metrics = [r_loss, oe, coverage_error_val,
               average_auprc, weighted_auprc,
               micro_f1, micro_recall, micro_precision,
               macro_f1, macro_recall, macro_precision,
               subset_accuracy, hl,
               ml_f_one, ml_recall, ml_precision]

    dict_metrics = {name: metric for name, metric in zip(metric_names, metrics)}
    df = pd.DataFrame(dict_metrics).T 

    return df


def info_message(message, *args, end="\n"):
    print(message.format(*args), end=end)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path) 
    return path

def create_store_path(*args):
    return "/".join(args)

def store_results(df, store_path, text_file = "metrics.txt"):
    df.index.name = "metrics"
    os.makedirs(store_path, exist_ok=True)
    df.to_csv(f"{store_path}/{text_file}", sep="\t")