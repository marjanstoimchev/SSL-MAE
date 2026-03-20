import copy
import torch
import lightning as L
from torch import Tensor
from typing import Any
from .schedulers import CosineWarmupScheduler


class MAELearner(L.LightningModule):
    """Lightning Module for SSL-MAE.

    Batch formats:
    - supervised/baseline: (x, x_raw, mask, non_masked, y, idx) — 6 elements
    - semi_supervised: (u_x, u_xr, u_m, u_nm, u_y, u_i, l_x, l_xr, l_m, l_nm, l_y, l_i) — 12 elements
      Concatenated to (2N, ...) with labeled portion having real labels, unlabeled having -1 sentinel.
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        ema_decay = config.training.get('ema_decay', 0.0)
        use_ema = ema_decay > 0 and config.training.mode != "supervised_baseline"
        if use_ema:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.requires_grad_(False)
            self.ema_decay = ema_decay
        else:
            self.ema_model = None
            self.ema_decay = 0.0

    @property
    def is_baseline(self):
        return self.config.training.mode == "supervised_baseline"

    @property
    def is_semi(self):
        return self.config.training.mode == "semi_supervised"

    def _unpack_batch(self, batch):
        """Unpack batch. Semi-supervised: concatenate unlabeled + labeled."""
        if len(batch) == 12:
            u_x, u_xr, u_m, u_nm, u_y, u_i, l_x, l_xr, l_m, l_nm, l_y, l_i = batch
            x = torch.cat([u_x, l_x], dim=0)
            x_raw = torch.cat([u_xr, l_xr], dim=0)
            mask = torch.cat([u_m, l_m], dim=0)
            non_masked = torch.cat([u_nm, l_nm], dim=0)
            y = torch.cat([u_y, l_y], dim=0)  # unlabeled has -1 sentinel
            idx = torch.cat([u_i, l_i], dim=0) if isinstance(u_i, torch.Tensor) else None
            return x, x_raw, mask, non_masked, y, idx
        return batch  # 6 elements

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)

    def _eval_model(self):
        return self.ema_model if self.ema_model is not None else self.model

    # --- Training ---

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        x, x_raw, mask, non_masked, y, idx = self._unpack_batch(batch)

        if self.is_baseline:
            return self._baseline_step(x, mask, non_masked, y, "train")

        outputs = self.model(x, x_raw, mask, non_masked, y)
        loss = outputs['loss']

        log_dict = {'loss': loss, 'r_loss': outputs['r_loss']}
        c_loss = outputs.get('c_loss')
        if c_loss is not None and c_loss != 0:
            log_dict['c_loss'] = c_loss
        log_dict['w'] = outputs['w']

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True, batch_size=x.size(0))
        self._update_ema()
        return loss

    def _baseline_step(self, x, mask, non_masked, y, prefix):
        z = self.model.forward_features(x, mask)
        if non_masked is not None:
            nm_exp = non_masked.unsqueeze(-1).expand(-1, -1, self.model.hidden_dim)
            z = torch.gather(z, 1, nm_exp)
        logits = self.model.fc(z.mean(1))
        y_target = y.float() if self.config.data.learning_task == "mlc" else y
        loss = self.model.criterion(logits, y_target)
        key = "loss" if prefix == "train" else "val_loss"
        self.log(key, loss, on_step=(prefix == "train"), on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=x.size(0))
        return loss if prefix == "train" else logits

    # --- Validation (always 6-element batch from TransformWrapper) ---

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, x_raw, mask, non_masked, y, idx = batch
        m = self._eval_model()

        if self.is_baseline:
            z = m.forward_features(x, mask)
            if non_masked is not None:
                nm_exp = non_masked.unsqueeze(-1).expand(-1, -1, m.hidden_dim)
                z = torch.gather(z, 1, nm_exp)
            logits = m.fc(z.mean(1))
            y_target = y.float() if self.config.data.learning_task == "mlc" else y
            val_loss = m.criterion(logits, y_target)
        else:
            outputs = m(x, x_raw, mask, non_masked, y)
            c_loss = outputs.get('c_loss')
            val_loss = c_loss if c_loss is not None and c_loss != 0 else outputs['loss']

        self.log_dict({'val_loss': val_loss}, on_step=False, on_epoch=True,
                      prog_bar=True, sync_dist=True, batch_size=x.size(0))

    # --- Prediction ---

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        x, x_raw, mask, non_masked, y, idx = batch
        m = self._eval_model()

        if self.is_baseline:
            z = m.forward_features(x, mask)
            if non_masked is not None:
                nm_exp = non_masked.unsqueeze(-1).expand(-1, -1, m.hidden_dim)
                z = torch.gather(z, 1, nm_exp)
            return m.fc(z.mean(1))

        return m(x, x_raw, mask, non_masked, y)['logits']

    # --- Optimizer ---

    def configure_optimizers(self):
        cfg = self.config.training

        w_params, base_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (w_params if 'w_param' in name else base_params).append(param)

        param_groups = [{'params': base_params}]
        if w_params:
            param_groups.append({'params': w_params, 'lr': cfg.get('lr_w', cfg.lr)})

        optimizer = torch.optim.AdamW(
            param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )

        optimizer_config = {"optimizer": optimizer}

        if cfg.apply_scheduler:
            self.trainer.fit_loop.setup_data()
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(total_steps * cfg.warmup_epochs / max(cfg.epochs, 1))
            min_lr_ratio = cfg.get('min_lr', 1e-5) / cfg.lr

            scheduler = CosineWarmupScheduler(
                optimizer, warmup_steps=warmup_steps,
                total_steps=total_steps, min_lr_ratio=min_lr_ratio,
            )
            optimizer_config["lr_scheduler"] = {
                "scheduler": scheduler, "name": "train/lr",
                "interval": "step", "frequency": 1,
            }

        return optimizer_config
