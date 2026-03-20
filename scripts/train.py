#!/usr/bin/env python
"""SSL-MAE Training Script.

Usage:
    python scripts/train.py --config configs/ucm_mlc.yaml
    python scripts/train.py --config configs/ucm_mlc.yaml training.epochs=50 data.batch_size=32
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor

from src.utils.config import load_config, parse_args
from src.utils.evaluate import evaluate
from src.models import ssl_mae, ssl_ijepa
from src.data.datamodule import SSLMAEDataModule
from src.trainers import MAELearner, IJEPALearner
from src.trainers.callbacks import ModelCheckpoint_, EarlyStopping_, RichProgressBar_


def create_logger(cfg):
    """Create logger based on config: wandb | tensorboard | csv."""
    fl = cfg.data.fraction_labeled
    save_dir = os.path.join(cfg.training.checkpoint.dir, cfg.experiment.name, cfg.training.mode, f"fl_{fl}")
    name = f"{cfg.experiment.name}_{cfg.training.mode}_fl{fl}"

    loggers = {
        "wandb": lambda: WandbLogger(project=cfg.logging.project, name=name, save_dir=save_dir),
        "tensorboard": lambda: TensorBoardLogger(save_dir=save_dir, name="logs"),
        "csv": lambda: CSVLogger(save_dir=save_dir, name="logs"),
    }

    logger_type = cfg.logging.logger
    if logger_type not in loggers:
        raise ValueError(f"Unknown logger '{logger_type}'. Choose from: {list(loggers.keys())}")
    return loggers[logger_type]()


def main():
    config_path, overrides = parse_args()
    cfg = load_config(config_path, cli_args=overrides)

    # semi_supervised with fl=1.0 is identical to supervised
    if cfg.training.mode == "semi_supervised" and cfg.data.fraction_labeled >= 1.0:
        from omegaconf import OmegaConf
        OmegaConf.update(cfg, "training.mode", "supervised")
        print("Note: fraction_labeled=1.0 with semi_supervised → using supervised mode")

    L.seed_everything(cfg.experiment.seed, workers=True)

    # Model
    model_type = cfg.model.get("type", "mae")
    model_factory = {"mae": ssl_mae, "ijepa": ssl_ijepa}
    if model_type not in model_factory:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(model_factory.keys())}")

    model_kwargs = dict(
        architecture=cfg.model.architecture,
        model_size=cfg.model.model_size,
        learning_task=cfg.data.learning_task,
        n_classes=cfg.data.n_classes,
        w=cfg.model.w,
    )
    if model_type == "ijepa":
        model_kwargs.update(
            predictor_embed_dim=cfg.model.get("predictor_embed_dim", 384),
            predictor_depth=cfg.model.get("predictor_depth", 6),
            ema_decay=cfg.model.get("ema_target_decay", 0.996),
        )
    model = model_factory[model_type](**model_kwargs)

    # DataModule
    datamodule = SSLMAEDataModule(cfg)

    # Lightning Module
    learner_cls = IJEPALearner if model_type == "ijepa" else MAELearner
    lightning_model = learner_cls(model, cfg)

    # Strategy
    devices = list(cfg.training.devices)
    if len(devices) > 1:
        strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=False)
    else:
        strategy = cfg.training.get("strategy", "auto")

    # Logger
    logger = create_logger(cfg)

    # Checkpoint dir: outputs/{name}/{mode}/fl_{fraction_labeled}/
    fl = cfg.data.fraction_labeled
    dirpath = os.path.join(
        cfg.training.checkpoint.dir, cfg.experiment.name,
        cfg.training.mode, f"fl_{fl}",
    )

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar_(),
        EarlyStopping_(
            metric=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            patience=cfg.training.patience,
        ),
    ]

    # Trainer
    trainer = L.Trainer(
        enable_model_summary=True,
        num_sanity_val_steps=0,
        accelerator='auto',
        strategy=strategy,
        devices=devices,
        precision=cfg.training.precision,
        min_epochs=5,
        max_epochs=cfg.training.epochs,
        accumulate_grad_batches=cfg.training.n_accumulate,
        sync_batchnorm=cfg.training.sync_batchnorm,
        benchmark=True,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_checkpointing=False,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(lightning_model, datamodule=datamodule)

    # Save final model weights (prefer EMA if available)
    save_model = getattr(lightning_model, 'ema_model', None)
    if save_model is None:
        save_model = lightning_model.model
    save_path = os.path.join(dirpath, "final_model.pt")
    os.makedirs(dirpath, exist_ok=True)
    torch.save(save_model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Evaluate on test set
    evaluate(trainer, lightning_model, datamodule, cfg)


if __name__ == '__main__':
    main()
