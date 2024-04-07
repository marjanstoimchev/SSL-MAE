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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=3, verbose=False, delta=0, trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

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
            x = data['image'].to(device, dtype=torch.float)
            y = data['label'].to(device)
            logits = model(x, labels = y, mask_ratio = 0.0)['logits']
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

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, h * patch_size))
    return imgs
    
def patchify(imgs, patch_size = 16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0
    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size ** 2 * 3))
    return x

def random_masking(imgs, patch_size, mask_ratio, mode = "cnn"):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence 
    """
    x = patchify(imgs, patch_size)
    N, L, D = x.shape  # batch, length, dim

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    if mode == "cnn":
        x_masked = unpatchify(x * (1 - mask.unsqueeze(-1)), patch_size)
    elif mode == "transformer":
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    return x_masked, mask, ids_keep, ids_restore

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
