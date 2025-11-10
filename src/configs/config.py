import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict

# BigEarthNet labels and mappings #

original = {
        "Continuous urban fabric": 0,
        "Discontinuous urban fabric": 1,
        "Industrial or commercial units": 2,
        "Road and rail networks and associated land": 3,
        "Port areas": 4,
        "Airports": 5,
        "Mineral extraction sites": 6,
        "Dump sites": 7,
        "Construction sites": 8,
        "Green urban areas": 9,
        "Sport and leisure facilities": 10,
        "Non-irrigated arable land": 11,
        "Permanently irrigated land": 12,
        "Rice fields": 13,
        "Vineyards": 14,
        "Fruit trees and berry plantations": 15,
        "Olive groves": 16,
        "Pastures": 17,
        "Annual crops associated with permanent crops": 18,
        "Complex cultivation patterns": 19,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 20,
        "Agro-forestry areas": 21,
        "Broad-leaved forest": 22,
        "Coniferous forest": 23,
        "Mixed forest": 24,
        "Natural grassland": 25,
        "Moors and heathland": 26,
        "Sclerophyllous vegetation": 27,
        "Transitional woodland/shrub": 28,
        "Beaches, dunes, sands": 29,
        "Bare rock": 30,
        "Sparsely vegetated areas": 31,
        "Burnt areas": 32,
        "Inland marshes": 33,
        "Peatbogs": 34,
        "Salt marshes": 35,
        "Salines": 36,
        "Intertidal flats": 37,
        "Water courses": 38,
        "Water bodies": 39,
        "Coastal lagoons": 40,
        "Estuaries": 41,
        "Sea and ocean": 42
    }

non_existing = [3, 4, 5, 6, 7 ,8, 9, 10, 30, 32, 37]

labels_19 = {
        "Urban fabric": 0,
        "Industrial or commercial units": 1,
        "Arable land": 2,
        "Permanent crops": 3,
        "Pastures": 4,
        "Complex cultivation patterns": 5,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
        "Agro-forestry areas": 7,
        "Broad-leaved forest": 8,
        "Coniferous forest": 9,
        "Mixed forest": 10,
        "Natural grassland and sparsely vegetated areas": 11,
        "Moors, heathland and sclerophyllous vegetation": 12,
        "Transitional woodland, shrub": 13,
        "Beaches, dunes, sands": 14,
        "Inland wetlands": 15,
        "Coastal wetlands": 16,
        "Inland waters": 17,
        "Marine waters": 18
    }

new_mappings =  {0: [0, 1],
                 1: [2],
                 2: [11,12,13],
                 3: [14, 15, 16, 18],
                 4: [17],
                 5: [19],
                 6: [20],
                 7: [21],
                 8: [22],
                 9: [23],
                 10: [24],
                 11: [25, 31],
                 12: [26, 27],
                 13: [28],
                 14: [29],
                 15: [33, 34],
                 16: [35, 36],
                 17: [38, 39],
                 18: [40, 41, 42]
             
             }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@dataclass
class BaseConfig:
    num_workers: int = 4
    seed: int = 42
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 1e-6
    n_accumulate: int = 8
    verbose_step: int = 1
    patience: int = 10
    apply_scheduler: int = 1
    save_model_dir: str = 'saved_models'


@dataclass
class DatasetConfig:
    image_size: int
    n_classes: int
    extension: str

class ConfigSelector(BaseConfig):
    def __init__(self):
        super(ConfigSelector, self).__init__()
        self.args = self.ConfigParser()
        self.learning_task = self.args.learning_task
        self.dataset = self.args.dataset
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.architecture = self.args.architecture
        self.model_size = self.args.model_size
        self.seed = self.args.seed
        self.mask_ratio = self.args.mask_ratio
        self.mode = self.args.mode

        # Ensure the selected dataset exists within the learning task
        if self.dataset not in self.get_available_datasets(self.learning_task):
            raise ValueError(f"Dataset '{self.dataset}' doesn't exist within the '{self.learning_task}' learning task.")

        self.n_classes = self.get_dataset_config().n_classes
        self.image_size = self.get_dataset_config().image_size

    def get_available_datasets(self, learning_task):
        available_datasets = {
            "mlc": ["UCM_mlc", "AID_mlc", "Ankara", "DFC_15", "BEN_43", "MLRSNet"],
            "mcc": ["OPTIMAL-31", "UCM_mcc", "RSSCN7", "AID_mcc", "RESISC45"]
        }
        return available_datasets.get(learning_task, [])

    def get_dataset_config(self):
        datasets = {
            "mlc": {
                "UCM_mlc": DatasetConfig(image_size=224, n_classes=17, extension="tif"),
                "AID_mlc": DatasetConfig(image_size=224, n_classes=17, extension="jpg"),
                "DFC_15": DatasetConfig(image_size=224, n_classes=8, extension="png"),
                "BEN_43": DatasetConfig(image_size=224, n_classes=43, extension="tif"),
                "MLRSNet": DatasetConfig(image_size=224, n_classes=60, extension="jpg"),
            },
            "mcc": {
                "OPTIMAL-31": DatasetConfig(image_size=224, n_classes=31, extension="jpg"),
                "UCM_mcc": DatasetConfig(image_size=224, n_classes=21, extension="tif"),
                "RSSCN7": DatasetConfig(image_size=224, n_classes=7, extension="jpg"),
                "AID_mcc": DatasetConfig(image_size=224, n_classes=30, extension="jpg"),
                "RESISC45": DatasetConfig(image_size=224, n_classes=45, extension="jpg"),
            }
        }
        
        dataset_configs = datasets.get(self.learning_task, {})
        return dataset_configs.get(self.dataset, DatasetConfig(image_size=224, n_classes=0, extension="unknown"))

    def ConfigParser(self):
        parser = argparse.ArgumentParser(description='Process hyper-parameters')
        parser.add_argument('--learning_task', type=str, default="mlc", help='Dataset')
        parser.add_argument('--dataset', type=str, default="UCM_mlc", help='Dataset')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--mask_ratio', type=float, default=0.3, help='Mask ratio')
        parser.add_argument('--architecture', type=str, default="deit", help='Architecture')
        parser.add_argument('--model_size', type=str, default="base", help='Model size')
        parser.add_argument('--seed', type=int, default=0, help='Seed')
        parser.add_argument('--mode', type=str, default='semi_supervised', help='Training mode')
        args, unknown = parser.parse_known_args() 
        return args