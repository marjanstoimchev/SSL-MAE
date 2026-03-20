import torch
import lightning as L
from torch import Tensor
from typing import Any
from .schedulers import IJEPALRScheduler, CosineWDScheduler


class IJEPALearner(L.LightningModule):
    """Lightning Module for I-JEPA.

    Batch formats:
    - supervised/baseline: (x, x_raw, mask, non_masked, y, idx) — 6 elements
    - semi_supervised: 12 elements (unlabeled + labeled paired), concatenated to 2N
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.automatic_optimization = False
        self._lr_scheduler = None
        self._wd_scheduler = None

    @property
    def is_baseline(self):
        return self.config.training.mode == "supervised_baseline"

    def _unpack_batch(self, batch):
        """Unpack batch. Semi-supervised: concatenate unlabeled + labeled."""
        if len(batch) == 12:
            u_x, u_xr, u_m, u_nm, u_y, u_i, l_x, l_xr, l_m, l_nm, l_y, l_i = batch
            x = torch.cat([u_x, l_x], dim=0)
            x_raw = torch.cat([u_xr, l_xr], dim=0)
            mask = torch.cat([u_m, l_m], dim=0)
            non_masked = torch.cat([u_nm, l_nm], dim=0)
            y = torch.cat([u_y, l_y], dim=0)
            idx = torch.cat([u_i, l_i], dim=0) if isinstance(u_i, torch.Tensor) else None
            return x, x_raw, mask, non_masked, y, idx
        return batch

    # --- Training ---

    def training_step(self, batch: Any, batch_idx: int) -> None:
        x, x_raw, mask, non_masked, y, idx = self._unpack_batch(batch)
        opt = self.optimizers()

        if self.is_baseline:
            self._baseline_train(x, mask, non_masked, y, opt)
            return

        outputs = self.model(x, x_raw, mask, non_masked, y)
        loss = outputs['loss']

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        if self._wd_scheduler is not None:
            self._wd_scheduler.step()

        log_dict = {'loss': loss, 'r_loss': outputs['r_loss']}
        c_loss = outputs.get('c_loss')
        if c_loss is not None and c_loss != 0:
            log_dict['c_loss'] = c_loss
        log_dict['w'] = outputs['w']

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True,
                      logger=True, sync_dist=True, batch_size=x.size(0))

    def _baseline_train(self, x, mask, non_masked, y, opt):
        # I-JEPA encoder already returns only visible patches (no gather needed)
        z = self.model.forward_features(x, mask)
        logits = self.model.fc(z.mean(1))
        y_target = y.float() if self.config.data.learning_task == "mlc" else y
        loss = self.model.criterion(logits, y_target)

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        if self._wd_scheduler is not None:
            self._wd_scheduler.step()

        self.log('loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=x.size(0))

    # --- Validation (always 6-element batch) ---

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, x_raw, mask, non_masked, y, idx = batch

        if self.is_baseline:
            z = self.model.forward_features(x, mask)
            logits = self.model.fc(z.mean(1))
            y_target = y.float() if self.config.data.learning_task == "mlc" else y
            val_loss = self.model.criterion(logits, y_target)
        else:
            outputs = self.model(x, x_raw, mask, non_masked, y)
            c_loss = outputs.get('c_loss')
            val_loss = c_loss if c_loss is not None and c_loss != 0 else outputs['loss']

        self.log_dict({'val_loss': val_loss}, on_step=False, on_epoch=True,
                      prog_bar=True, sync_dist=True, batch_size=x.size(0))

    # --- Prediction ---

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        x, x_raw, mask, non_masked, y, idx = batch

        if self.is_baseline:
            z = self.model.forward_features(x, mask)
            return self.model.fc(z.mean(1))

        outputs = self.model(x, x_raw, mask, non_masked, y)
        return outputs.get('logits')

    # --- Optimizer ---

    def configure_optimizers(self):
        cfg = self.config.training

        encoder_wd, encoder_nowd = [], []
        predictor_wd, predictor_nowd = [], []
        w_params, cls_params = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'w_param' in name or name.startswith('w.'):
                w_params.append(param)
            elif name.startswith('fc.'):
                cls_params.append(param)
            elif name.startswith('predictor.'):
                if param.dim() < 2 or 'norm' in name or 'bias' in name:
                    predictor_nowd.append(param)
                else:
                    predictor_wd.append(param)
            else:
                if param.dim() < 2 or 'norm' in name or 'bias' in name:
                    encoder_nowd.append(param)
                else:
                    encoder_wd.append(param)

        param_groups = [
            {'params': encoder_wd},
            {'params': encoder_nowd, 'WD_exclude': True, 'weight_decay': 0.0},
            {'params': predictor_wd},
            {'params': predictor_nowd, 'WD_exclude': True, 'weight_decay': 0.0},
            {'params': cls_params},
        ]
        if w_params:
            param_groups.append({'params': w_params, 'lr': cfg.get('lr_w', cfg.lr)})

        optimizer = torch.optim.AdamW(
            param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )

        self.trainer.fit_loop.setup_data()
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * cfg.warmup_epochs / max(cfg.epochs, 1))

        self._lr_scheduler = IJEPALRScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            start_lr=cfg.get('start_lr', cfg.lr * 0.1),
            ref_lr=cfg.lr,
            final_lr=cfg.get('min_lr', 1e-6),
            total_steps=total_steps,
        )

        self._wd_scheduler = CosineWDScheduler(
            optimizer,
            ref_wd=cfg.weight_decay,
            final_wd=cfg.get('final_weight_decay', cfg.weight_decay),
            total_steps=total_steps,
        )

        return optimizer
