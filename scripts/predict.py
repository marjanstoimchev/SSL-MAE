#!/usr/bin/env python
"""SSL-MAE Prediction/Inference Script.

Usage:
    # Auto-find best checkpoint from output dir:
    python scripts/predict.py --config configs/ucm_mlc.yaml --checkpoint outputs/ucm_mlc/semi_supervised

    # With specific fraction (default uses config's fraction_labeled):
    python scripts/predict.py --config configs/ucm_mlc.yaml --checkpoint outputs/ucm_mlc/semi_supervised data.fraction_labeled=0.05

    # Exact checkpoint path:
    python scripts/predict.py --config configs/ucm_mlc.yaml --checkpoint outputs/ucm_mlc/semi_supervised/fl_0.1/best-epoch=12-val_loss=0.3421.ckpt
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lightning as L

from src.utils.config import load_config, parse_args
from src.utils.evaluate import evaluate
from src.models import ssl_mae, ssl_ijepa
from src.data.datamodule import SSLMAEDataModule
from src.trainers import MAELearner, IJEPALearner


def get_checkpoint_path():
    """Extract --checkpoint from sys.argv."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--checkpoint' and i + 1 < len(args):
            return args[i + 1]
    raise ValueError("Must provide --checkpoint <path_to_ckpt>")


def resolve_checkpoint(checkpoint_path, fraction_labeled):
    """Resolve checkpoint path — accepts exact .ckpt file or a directory.

    If a directory is given, looks for best-*.ckpt inside fl_{fraction_labeled}/.
    Falls back to last.ckpt if no best found.
    """
    # Exact checkpoint file
    if checkpoint_path.endswith('.ckpt') and os.path.isfile(checkpoint_path):
        return checkpoint_path

    # Directory — look inside fl_{fraction}/
    fl_dir = os.path.join(checkpoint_path, f"fl_{fraction_labeled}")
    if not os.path.isdir(fl_dir):
        # Maybe the fl_ dir is already in the path
        fl_dir = checkpoint_path

    # Try best checkpoint first
    best = sorted(glob.glob(os.path.join(fl_dir, "best-*.ckpt")))
    if best:
        print(f"Found best checkpoint: {best[-1]}")
        return best[-1]

    # Fall back to last.ckpt
    last = os.path.join(fl_dir, "last.ckpt")
    if os.path.isfile(last):
        print(f"Found last checkpoint: {last}")
        return last

    raise FileNotFoundError(
        f"No checkpoint found in {fl_dir}. "
        f"Looked for best-*.ckpt and last.ckpt"
    )


def main():
    config_path, overrides = parse_args()
    cfg = load_config(config_path, cli_args=overrides)
    checkpoint_input = get_checkpoint_path()

    L.seed_everything(cfg.experiment.seed, workers=True)

    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint(checkpoint_input, cfg.data.fraction_labeled)

    # Model
    model_type = cfg.model.get("type", "mae")
    model_factory = {"mae": ssl_mae, "ijepa": ssl_ijepa}
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

    # Load from checkpoint
    learner_cls = IJEPALearner if model_type == "ijepa" else MAELearner
    lightning_model = learner_cls.load_from_checkpoint(
        checkpoint_path, model=model, config=cfg,
    )

    # DataModule
    datamodule = SSLMAEDataModule(cfg)
    datamodule.setup()

    # Trainer (inference only)
    trainer = L.Trainer(
        enable_model_summary=False,
        accelerator='auto',
        devices=[0],
        precision=cfg.training.precision,
    )

    evaluate(trainer, lightning_model, datamodule, cfg)


if __name__ == '__main__':
    main()
