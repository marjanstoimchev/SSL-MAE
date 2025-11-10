import os, glob, gc
import numpy as np
import torch
from tqdm import tqdm
from utils.utils import calculate_mlc_metrics, calculate_mcc_metrics
from utils.dataset_utils import one_hot_encode
from configs.config import ConfigSelector
from dataclasses import dataclass
from typing import Any, Callable, Dict
from utils.utils import info_message

class GeneratePredictions(ConfigSelector):
    def __init__(self):
        super(GeneratePredictions, self).__init__()
        
    @torch.inference_mode()
    def predict_mlc(self, model, dataloader, device):
        model.eval()
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Making predictions for ---test set ---: ", position=0, leave=True)
        
        probas, targets = [], []
        
        for _, data in pbar:
            # x = data['image'].to(device, dtype=torch.float)
            # y = data['label'].to(device)
            # logits = model(x, labels = y, mask_ratio = 0.0)['logits']
            x, mask, y = data
            x = x.to(device, dtype=torch.float)
            y = y.to(device)
            mask = mask.to(device)
            logits = model(x, labels = y, mask = mask)['logits']

            proba = torch.sigmoid(logits).data
            probas += [proba.detach().to("cpu")]
            targets += [y.detach().to("cpu")]
        
        targets = torch.cat(targets, dim=0).to(torch.int).numpy()
        probas = torch.cat(probas, dim=0).numpy()
        preds = (probas > 0.5).astype('int')
        
        # detect all zeros in y_target #
        where_not_zero = targets.any(axis=1)
        targets = targets[where_not_zero]
        probas  = probas[where_not_zero]
        preds   = preds[where_not_zero]

        Y = dict()
        Y['y_scores'] = probas
        Y['y_pred']   = preds
        Y['y_true']   = targets
        info_message("")
        info_message("shape targets: {}", Y['y_true'].shape)
        info_message("shape scores: {}", Y['y_scores'].shape)
        info_message("shape predictions : {}", Y['y_pred'].shape)
        info_message("")
        return Y
    
    @torch.inference_mode()
    def predict_mcc(self, model, dataloader, device):
        model.eval()
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Making predictions for ---test set ---: ", position=0, leave=True)
        
        probas, targets, targets_one_hot = [], [], []
        
        for _, data in pbar:
            x = data['image'].to(device, dtype=torch.float)
            y = data['label'].to(device)
            y_h = data['one_hot'].to(device)

            logits = model(x, labels = y, mask_ratio = 0.0)['logits']
            proba = torch.softmax(logits, dim = 1).data
            probas += [proba.detach().to("cpu")]
            targets += [y.detach().to("cpu")]
            targets_one_hot += [y_h.detach().to("cpu")]
        
        targets = torch.cat(targets, dim=0).to(torch.int).numpy()
        targets_one_hot = torch.cat(targets_one_hot, dim=0).to(torch.int).numpy()
        probas  = torch.cat(probas, dim=0).numpy()
        preds   = probas.argmax(1).astype('int')
        
        Y = dict()
        Y['y_scores'] = probas
        Y['y_pred']   = preds
        Y['y_true']   = targets
        Y['one_hot'] = targets_one_hot

        info_message("")
        info_message("shape targets: {}", Y['y_true'].shape)
        info_message("shape targets one-hot: {}", Y['one_hot'].shape)
        info_message("shape scores: {}", Y['y_scores'].shape)
        info_message("shape predictions : {}", Y['y_pred'].shape)
        info_message("")
        return Y

    def run_prediction(self, model, dataloader, device):  
        info_message("\n")
        info_message("MODEL PREDICTION...")
        info_message("\n")
    
        if self.learning_task == "mlc":
            predict = self.predict_mlc
            calculate_metrics = calculate_mlc_metrics
        elif self.learning_task == "mcc":
            predict = self.predict_mcc
            calculate_metrics = calculate_mcc_metrics 
        
        Y = predict(model, dataloader, device)
        metrics = calculate_metrics(Y)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return metrics

def load_torch_model(backbone, state_dict_name, dir_path = "saved_models", ext = "pth"):
    PATH = glob.glob(os.path.join(dir_path, f'**/*.{ext}'), recursive=True)[0]
    print(f"Loading model from path: {PATH}")
    checkpoint = torch.load(PATH)
    state_dict = checkpoint[state_dict_name]
    backbone.load_state_dict(state_dict)
    backbone.eval()
    print("\nDone !")
    return backbone

def run_torch_inference(dirpath, state_dict_name, backbone, dataloader, device):
    inference_model = load_torch_model(
        backbone=backbone,
        state_dict_name = state_dict_name,
        dir_path=dirpath, 
        )
    
    predictor = GeneratePredictions()
    df_metrics = predictor.run_prediction(inference_model, dataloader, device)
    return df_metrics


def get_optimizer(model, lr):
    my_list = ['w_param.w_param']
    params_w = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    base_p = filter(lambda p: p.requires_grad, base_params) 
    lr_w = 1e-3
    optimizer = torch.optim.AdamW([
        {'params': base_p}, 
        {'params': params_w, 'lr': lr_w}
        ], lr=lr, betas=(0.9, 0.95))
    
    return optimizer
