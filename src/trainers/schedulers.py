"""Learning rate and weight decay schedulers."""

import math
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """MAE-style: warmup from 0 to base_lr, then cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class IJEPALRScheduler:
    """I-JEPA style: warmup from start_lr to ref_lr, then cosine decay to final_lr.

    Not a torch scheduler — call .step() manually each iteration.
    From the original I-JEPA implementation.
    """

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, final_lr, total_steps):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = total_steps - warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = self._step / max(1, self.warmup_steps)
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.T_max)
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
        return new_lr


class CosineWDScheduler:
    """I-JEPA style: cosine weight decay schedule from ref_wd to final_wd.

    From the original I-JEPA implementation.
    """

    def __init__(self, optimizer, ref_wd, final_wd, total_steps):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = total_steps
        self._step = 0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1 + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if not group.get('WD_exclude', False):
                group['weight_decay'] = new_wd
        return new_wd
