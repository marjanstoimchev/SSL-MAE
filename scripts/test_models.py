#!/usr/bin/env python
"""Rigorous tests for MAE and I-JEPA models + learners."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

def make_batch(B=4, labeled_frac=1.0, n_classes=17, mask_ratio=0.75):
    """Create a test batch: (x, x_raw, mask, non_masked, y, idx)."""
    x = torch.randn(B, 3, 224, 224)
    x_raw = torch.rand(B, 3, 224, 224)
    num_patches = 196
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(B, num_patches, dtype=torch.long)
    mask[:, :num_masked] = 1
    non_masked = torch.arange(num_masked, num_patches).unsqueeze(0).expand(B, -1)
    y = torch.rand(B, n_classes)

    # Mark some as unlabeled
    n_labeled = max(1, int(B * labeled_frac))
    y[n_labeled:] = -1

    idx = torch.arange(B)
    return x, x_raw, mask, non_masked, y, idx


def test_mae_model():
    print("=" * 60)
    print("TEST: MAE Model")
    print("=" * 60)
    from src.models import ssl_mae

    model = ssl_mae('vit', 'small', 'mlc', 17, None)
    x, x_raw, mask, non_masked, y, idx = make_batch()

    # Full forward
    out = model(x, x_raw, mask, non_masked, y)
    assert 'loss' in out and 'r_loss' in out and 'w' in out
    assert 'logits' in out  # has labeled samples
    assert 'x_rec' in out   # MAE reconstructs pixels
    assert out['loss'].requires_grad
    print("  forward:       OK")

    # Forward without labels
    out2 = model(x, x_raw, mask, non_masked, None)
    assert out2['c_loss'] == 0
    print("  no labels:     OK")

    # Forward with all unlabeled
    y_unlabeled = torch.full((4, 17), -1.0)
    out3 = model(x, x_raw, mask, non_masked, y_unlabeled)
    assert out3.get('logits') is None or 'logits' not in out3
    print("  all unlabeled: OK")

    # Denormalize reconstruction
    rec = model.denormalize_reconstruction(out['x_rec'], x_raw)
    assert rec.shape == x_raw.shape
    assert rec.min() >= 0 and rec.max() <= 1
    print("  denorm recon:  OK")

    # Backward
    out['loss'].backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0
             for p in model.parameters() if p.requires_grad]
    assert any(grads)
    print("  backward:      OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"  params:        {total/1e6:.1f}M")
    print()


def test_ijepa_model():
    print("=" * 60)
    print("TEST: I-JEPA Model")
    print("=" * 60)
    from src.models import ssl_ijepa

    model = ssl_ijepa('vit', 'small', 'mlc', 17, None)
    x, x_raw, mask, non_masked, y, idx = make_batch()

    # Full forward
    out = model(x, x_raw, mask, non_masked, y)
    assert 'loss' in out and 'r_loss' in out and 'w' in out
    assert 'logits' in out
    assert 'x_rec' not in out  # I-JEPA does NOT reconstruct pixels
    assert out['loss'].requires_grad
    print("  forward:       OK")

    # Check EMA target encoder not trainable
    for p in model.target_encoder.parameters():
        assert not p.requires_grad
    print("  target frozen: OK")

    # Forward without labels
    out2 = model(x, x_raw, mask, non_masked, None)
    assert out2.get('c_loss', 0) == 0
    print("  no labels:     OK")

    # EMA update
    old_target = model.target_encoder.blocks[0].attn.qkv.weight.data.clone()
    model.update_target_encoder()
    new_target = model.target_encoder.blocks[0].attn.qkv.weight.data
    assert not torch.equal(old_target, new_target) or model.ema_decay == 1.0
    print("  EMA update:    OK")

    # No EMA during eval
    model.eval()
    target_before = model.target_encoder.blocks[0].attn.qkv.weight.data.clone()
    _ = model(x, x_raw, mask, non_masked, y)
    target_after = model.target_encoder.blocks[0].attn.qkv.weight.data
    assert torch.equal(target_before, target_after)
    model.train()
    print("  no EMA eval:   OK")

    # Backward
    model.zero_grad()
    out3 = model(x, x_raw, mask, non_masked, y)
    out3['loss'].backward()
    encoder_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.encoder.parameters() if p.requires_grad)
    predictor_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.predictor.parameters() if p.requires_grad)
    assert encoder_grads, "Encoder should have gradients"
    assert predictor_grads, "Predictor should have gradients"
    print("  backward:      OK")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params:        {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print()


def test_mae_learner():
    print("=" * 60)
    print("TEST: MAE Learner")
    print("=" * 60)
    from src.models import ssl_mae
    from src.trainers import MAELearner
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        'training': {'mode': 'semi_supervised', 'lr': 1e-4, 'lr_w': 1e-3,
                     'weight_decay': 1e-6, 'warmup_epochs': 5, 'epochs': 100,
                     'apply_scheduler': False, 'ema_decay': 0},
        'data': {'learning_task': 'mlc'},
    })

    model = ssl_mae('vit', 'small', 'mlc', 17, None)
    learner = MAELearner(model, cfg)

    batch = make_batch(labeled_frac=0.5)
    loss = learner.training_step(batch, 0)
    assert loss.requires_grad
    print("  train step:    OK")

    learner.eval()
    learner.validation_step(batch, 0)
    print("  val step:      OK")

    with torch.no_grad():
        logits = learner.predict_step(batch, 0)
    assert logits is not None
    print("  predict step:  OK")

    # Baseline mode
    cfg.training.mode = 'supervised_baseline'
    learner2 = MAELearner(model, cfg)
    loss2 = learner2.training_step(batch, 0)
    assert loss2.requires_grad
    print("  baseline:      OK")
    print()


def test_ijepa_learner():
    print("=" * 60)
    print("TEST: I-JEPA Learner")
    print("=" * 60)
    from src.models import ssl_ijepa
    from src.trainers import IJEPALearner
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        'training': {'mode': 'semi_supervised', 'lr': 1e-4, 'lr_w': 1e-3,
                     'weight_decay': 1e-6, 'warmup_epochs': 5, 'epochs': 100,
                     'apply_scheduler': False},
        'data': {'learning_task': 'mlc'},
    })

    model = ssl_ijepa('vit', 'small', 'mlc', 17, None)
    learner = IJEPALearner(model, cfg)

    # IJEPALearner uses manual optimization (needs trainer for optimizers())
    # Test val/predict which don't need optimizer
    batch = make_batch(labeled_frac=0.5)

    learner.eval()
    learner.validation_step(batch, 0)
    print("  val step:      OK")

    with torch.no_grad():
        logits = learner.predict_step(batch, 0)
    assert logits is not None
    print("  predict step:  OK")

    # Test model forward directly (training is tested via integration)
    model.train()
    x, x_raw, mask, non_masked, y, idx = batch
    out = model(x, x_raw, mask, non_masked, y)
    assert out['loss'].requires_grad
    out['loss'].backward()
    print("  model fwd/bwd: OK")

    # Verify param groups would be correct
    enc_p = [n for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('predictor.', 'fc.', 'w.'))]
    pred_p = [n for n, p in model.named_parameters() if p.requires_grad and n.startswith('predictor.')]
    assert len(enc_p) > 0, "No encoder params"
    assert len(pred_p) > 0, "No predictor params"
    print("  param groups:  OK")
    print()


def test_interface_compatibility():
    print("=" * 60)
    print("TEST: Interface Compatibility (MAE vs I-JEPA)")
    print("=" * 60)
    from src.models import ssl_mae, ssl_ijepa

    mae = ssl_mae('vit', 'small', 'mlc', 17, 0.5)
    ijepa = ssl_ijepa('vit', 'small', 'mlc', 17, 0.5)

    # Both must have these attributes
    for attr in ['hidden_dim', 'embed_dim', 'fc', 'criterion', 'w']:
        assert hasattr(mae, attr), f"MAE missing {attr}"
        assert hasattr(ijepa, attr), f"IJEPA missing {attr}"
    print("  shared attrs:  OK")

    # Both must accept same forward signature
    batch = make_batch()
    x, x_raw, mask, non_masked, y, idx = batch
    out_mae = mae(x, x_raw, mask, non_masked, y)
    out_ijepa = ijepa(x, x_raw, mask, non_masked, y)

    # Both must return same keys
    required_keys = {'loss', 'r_loss', 'c_loss', 'w', 'mask'}
    assert required_keys.issubset(out_mae.keys()), f"MAE missing: {required_keys - out_mae.keys()}"
    assert required_keys.issubset(out_ijepa.keys()), f"IJEPA missing: {required_keys - out_ijepa.keys()}"
    print("  output keys:   OK")

    # Both forward_features must work
    z_mae = mae.forward_features(x, mask)
    z_ijepa = ijepa.forward_features(x, mask)
    assert z_mae.dim() == 3  # (B, N, D)
    assert z_ijepa.dim() == 3
    print("  fwd_features:  OK")
    print()


def test_data_pipeline():
    print("=" * 60)
    print("TEST: Data Pipeline")
    print("=" * 60)
    from src.utils.config import load_config
    from src.data.datamodule import SSLMAEDataModule
    from omegaconf import OmegaConf

    cfg = load_config('configs/mae/ucm_mlc.yaml', cli_args=[])
    OmegaConf.update(cfg, 'data.batch_size', 4)
    OmegaConf.update(cfg, 'data.num_workers', 0)
    OmegaConf.update(cfg, 'data.fraction_labeled', 0.1)

    for mode in ['semi_supervised', 'supervised', 'supervised_baseline']:
        OmegaConf.update(cfg, 'training.mode', mode)
        dm = SSLMAEDataModule(cfg)
        dm.setup()

        # Check train batch format
        batch = next(iter(dm.train_dataloader()))
        assert len(batch) == 6, f"{mode}: batch should have 6 elements, got {len(batch)}"
        x, x_raw, mask, non_masked, y, idx = batch
        assert x.dim() == 4  # (B, 3, H, W)
        assert x_raw.dim() == 4
        assert mask.dim() == 2  # (B, N)
        assert non_masked.dim() == 2
        print(f"  {mode:25s} train: OK (n={len(dm.train_dataset)})")

        # Check val/test
        val_batch = next(iter(dm.val_dataloader()))
        assert len(val_batch) == 6
        test_batch = next(iter(dm.predict_dataloader()))
        assert len(test_batch) == 6
        print(f"  {mode:25s} val/test: OK")

    print()


def test_evaluate():
    print("=" * 60)
    print("TEST: Evaluate Pipeline")
    print("=" * 60)
    from src.utils.evaluate import collect_targets

    # Mock dataloader
    class MockDL:
        def __iter__(self):
            for _ in range(3):
                yield (torch.randn(4, 3, 224, 224),  # x
                       torch.rand(4, 3, 224, 224),    # x_raw
                       torch.zeros(4, 196),            # mask
                       torch.arange(196).unsqueeze(0).expand(4, -1),  # non_masked
                       torch.rand(4, 17),              # y
                       torch.arange(4))                # idx

    targets = collect_targets(MockDL())
    assert targets.shape == (12, 17)
    assert (targets >= 0).all()
    print("  collect_targets: OK")
    print()


if __name__ == '__main__':
    print()
    test_mae_model()
    test_ijepa_model()
    test_mae_learner()
    test_ijepa_learner()
    test_interface_compatibility()
    test_evaluate()
    test_data_pipeline()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
