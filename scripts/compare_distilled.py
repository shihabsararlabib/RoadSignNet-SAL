#!/usr/bin/env python3
"""Compare baseline and distilled checkpoints on the test set."""

import sys
import os
from pathlib import Path
import argparse
import yaml
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate import evaluate
from roadsignnet_sal.model import create_roadsignnet_transfer, create_roadsignnet_sal


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def smoke_test():
    """Lightweight smoke test without dataset."""
    device = torch.device('cpu')
    baseline = create_roadsignnet_sal(num_classes=43, width_multiplier=1.0).to(device).eval()
    distilled = create_roadsignnet_transfer(
        num_classes=43,
        backbone='efficientnet_b0+vit_tiny_patch16_224',
        pretrained=False
    ).to(device).eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _ = baseline(x)
        _ = distilled(x)

    print("Smoke OK")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs distilled checkpoints')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--baseline', type=str, help='Path to baseline checkpoint')
    parser.add_argument('--distilled', type=str, help='Path to distilled checkpoint')
    parser.add_argument('--baseline-backbone', type=str, default='mobilenet_v3_small')
    parser.add_argument('--distilled-backbone', type=str, default='mobilenet_v3_small')
    parser.add_argument('--smoke', action='store_true', help='Run a fast smoke test without dataset')
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    if not args.baseline or not args.distilled:
        raise ValueError('Both --baseline and --distilled checkpoints are required (or use --smoke).')

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = Path(__file__).parent.parent / config_path

    config = load_config(config_path)

    print("\n=== Evaluating baseline ===")
    base_results = evaluate(config, args.baseline, args.baseline_backbone)

    print("\n=== Evaluating distilled ===")
    dist_results = evaluate(config, args.distilled, args.distilled_backbone)

    # Compute deltas
    delta = {
        'baseline_checkpoint': args.baseline,
        'distilled_checkpoint': args.distilled,
        'delta_map@0.5': dist_results.get('mAP@0.5', 0.0) - base_results.get('mAP@0.5', 0.0),
        'delta_precision': dist_results.get('precision', 0.0) - base_results.get('precision', 0.0),
        'delta_recall': dist_results.get('recall', 0.0) - base_results.get('recall', 0.0),
        'delta_f1': dist_results.get('f1_score', 0.0) - base_results.get('f1_score', 0.0),
    }

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'compare_distilled_results.csv'

    df = pd.DataFrame([{
        **{f'base_{k}': v for k, v in base_results.items()},
        **{f'dist_{k}': v for k, v in dist_results.items()},
        **delta
    }])
    df.to_csv(out_path, index=False)

    print("\n=== Summary ===")
    print(f"mAP@0.5 delta: {delta['delta_map@0.5']:.4f}")
    print(f"Precision delta: {delta['delta_precision']:.4f}")
    print(f"Recall delta: {delta['delta_recall']:.4f}")
    print(f"F1 delta: {delta['delta_f1']:.4f}")
    print(f"Results saved: {out_path}")


if __name__ == '__main__':
    main()
