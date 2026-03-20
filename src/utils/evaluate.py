import os

import numpy as np
import torch
import lightning as L

from src.utils.metrics import calculate_mlc_metrics, calculate_mcc_metrics
from src.utils.helpers import store_results


def collect_targets(dataloader):
    """Collect all targets from a dataloader efficiently."""
    targets = []
    for batch in dataloader:
        # Batch format: (x, x_raw, mask, non_masked, y, idx)
        y = batch[4]
        targets.append(y)
    return torch.cat(targets, dim=0).cpu()


def evaluate(trainer, lightning_model, datamodule, cfg, results_dir=None):
    """Run prediction on test set and compute metrics.

    Args:
        trainer: Lightning Trainer instance.
        lightning_model: trained SSLMAE_Learner.
        datamodule: SSLMAEDataModule (must have setup() already called).
        cfg: OmegaConf config.
        results_dir: where to save metrics.txt. If None, auto-generates path.

    Returns:
        pd.DataFrame with evaluation metrics.
    """
    # Run predictions
    logits_list = trainer.predict(lightning_model, datamodule=datamodule)
    logits = torch.cat(logits_list, dim=0).cpu().float()

    # Collect targets
    targets = collect_targets(datamodule.predict_dataloader())

    # Compute metrics
    if cfg.data.learning_task == "mlc":
        probas = torch.sigmoid(logits).numpy()
        preds = (probas > 0.5).astype(int)
        targets_np = targets.numpy()

        # Remove all-zero targets
        nonzero = targets_np.any(axis=1)
        targets_np, probas, preds = targets_np[nonzero], probas[nonzero], preds[nonzero]

        Y = {'y_true': targets_np, 'y_pred': preds, 'y_scores': probas}
        df = calculate_mlc_metrics(Y)
    else:
        probas = torch.softmax(logits, dim=1).numpy()
        preds = probas.argmax(1).astype(int)
        targets_np = targets.numpy().astype(int)
        one_hot = np.eye(cfg.data.n_classes)[targets_np]

        Y = {'y_true': targets_np, 'y_pred': preds, 'y_scores': probas, 'one_hot': one_hot}
        df = calculate_mcc_metrics(Y)

    print("\n--- Evaluation Metrics ---")
    print(df)

    # Save results
    if results_dir is None:
        fl = cfg.data.fraction_labeled
        results_dir = os.path.join(
            cfg.training.checkpoint.dir, cfg.experiment.name,
            cfg.training.mode, f"fl_{fl}", "results",
        )
    store_results(df, results_dir)
    print(f"\nResults saved to {results_dir}/metrics.txt")

    return df
