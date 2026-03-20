#!/usr/bin/env python
"""SSL-MAE Sweep Script — run training across multiple fractions/modes/configs.

Usage:
    # Sweep fractions for one mode:
    python scripts/sweep.py --config configs/ucm_mlc.yaml \
        --fractions 0.01 0.05 0.1 0.2 0.5 1.0 \
        training.mode=semi_supervised training.epochs=100 training.devices=[1] data.batch_size=32

    # Sweep fractions for multiple modes:
    python scripts/sweep.py --config configs/ucm_mlc.yaml \
        --fractions 0.01 0.05 0.1 \
        --modes semi_supervised supervised_baseline \
        training.epochs=100 training.devices=[1]

    # Sweep mask ratios:
    python scripts/sweep.py --config configs/ucm_mlc.yaml \
        --fractions 0.1 \
        --mask_ratios 0.3 0.5 0.75 \
        training.mode=semi_supervised training.epochs=100 training.devices=[1]
"""

import os
import sys
import subprocess
import itertools
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_sweep_args():
    """Parse sweep-specific args and pass the rest to train.py."""
    args = sys.argv[1:]
    config_path = None
    fractions = None
    modes = None
    mask_ratios = None
    train_overrides = []

    i = 0
    while i < len(args):
        if args[i] == '--config' and i + 1 < len(args):
            config_path = args[i + 1]
            i += 2
        elif args[i] == '--fractions':
            fractions = []
            i += 1
            while i < len(args) and not args[i].startswith('--') and '=' not in args[i]:
                fractions.append(float(args[i]))
                i += 1
        elif args[i] == '--modes':
            modes = []
            i += 1
            while i < len(args) and not args[i].startswith('--') and '=' not in args[i]:
                modes.append(args[i])
                i += 1
        elif args[i] == '--mask_ratios':
            mask_ratios = []
            i += 1
            while i < len(args) and not args[i].startswith('--') and '=' not in args[i]:
                mask_ratios.append(float(args[i]))
                i += 1
        elif '=' in args[i]:
            train_overrides.append(args[i])
            i += 1
        else:
            i += 1

    if config_path is None:
        raise ValueError("Must provide --config <path>")
    if fractions is None:
        fractions = [0.1]

    return config_path, fractions, modes, mask_ratios, train_overrides


def build_runs(config_path, fractions, modes, mask_ratios, train_overrides):
    """Build list of (description, command) tuples."""
    # Extract mode from overrides if not in --modes
    override_mode = None
    base_overrides = []
    for o in train_overrides:
        if o.startswith('training.mode='):
            override_mode = o.split('=', 1)[1]
        else:
            base_overrides.append(o)

    if modes is None:
        modes = [override_mode or 'semi_supervised']

    if mask_ratios is None:
        mask_ratios = [None]

    runs = []
    seen = set()
    for mode, frac, mr in itertools.product(modes, fractions, mask_ratios):
        # semi_supervised with fl=1.0 is identical to supervised — use supervised
        effective_mode = mode
        if mode == "semi_supervised" and frac >= 1.0:
            effective_mode = "supervised"

        # Deduplicate (e.g. semi_supervised fl=1.0 and supervised fl=1.0)
        run_key = (effective_mode, frac, mr)
        if run_key in seen:
            continue
        seen.add(run_key)

        overrides = base_overrides + [
            f'training.mode={effective_mode}',
            f'data.fraction_labeled={frac}',
        ]
        if mr is not None:
            overrides.append(f'data.mask_ratio={mr}')

        desc = f"{effective_mode} | fl={frac}"
        if mr is not None:
            desc += f" | mr={mr}"

        cmd = [sys.executable, 'scripts/train.py', '--config', config_path] + overrides
        runs.append((desc, cmd))

    return runs


def main():
    config_path, fractions, modes, mask_ratios, train_overrides = parse_sweep_args()
    runs = build_runs(config_path, fractions, modes, mask_ratios, train_overrides)

    print(f"\n{'='*60}")
    print(f"  SSL-MAE Sweep — {len(runs)} run(s)")
    print(f"{'='*60}")
    for i, (desc, _) in enumerate(runs):
        print(f"  [{i+1}] {desc}")
    print(f"{'='*60}\n")

    for i, (desc, cmd) in enumerate(runs):
        print(f"\n{'─'*60}")
        print(f"  Run {i+1}/{len(runs)}: {desc}")
        print(f"  {' '.join(cmd)}")
        print(f"{'─'*60}\n")

        t0 = datetime.now()
        result = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'))
        elapsed = datetime.now() - t0

        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"\n  [{status}] {desc} — {elapsed}")

        if result.returncode != 0:
            print(f"  Skipping remaining runs due to failure.")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print(f"  All {len(runs)} runs completed.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
