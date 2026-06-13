#!/usr/bin/env python3
"""Run every method for both beta=10 (homogeneous) and beta=0.1 (heterogeneous),
then plot the Figure-1-style comparison.

Results are cached per (method, beta) in results/<method>.json, so re-running
this script after an interruption skips combinations that are already done.

Usage:
    python run_all.py                 # all 5 methods x 2 betas
    python run_all.py --methods fed_muon local_muon --betas 0.1
    python run_all.py --rounds 50     # quick smoke test
"""

import argparse
import os

from src.config import CONFIG
from src.data import get_datasets
from src.methods import METHODS
from src.plot import plot_comparison
from src.utils import get_device, load_json, save_json, set_seed
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                         choices=sorted(METHODS.keys()))
    parser.add_argument("--betas", nargs="+", type=float, default=CONFIG["beta_vals"])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(CONFIG)
    if args.rounds is not None:
        cfg["rounds"] = args.rounds
    seed = args.seed if args.seed is not None else cfg["seeds"][0]

    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    train_ds, test_ds = get_datasets(cfg["data_dir"])
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    for method in args.methods:
        out_path = os.path.join(cfg["results_dir"], f"{method}.json")
        all_results = load_json(out_path)

        for beta in args.betas:
            key = str(beta)
            if key in all_results and len(all_results[key]) >= cfg["rounds"]:
                print(f"[{method}] beta={beta} already complete -- skipping")
                continue

            print(f"\n{'=' * 60}\n{method}  beta={beta}\n{'=' * 60}")
            history = METHODS[method](train_ds, test_loader, beta, cfg, seed, device)
            all_results[key] = history
            save_json(all_results, out_path)
            print(f"  final={history[-1] * 100:.2f}%  best={max(history) * 100:.2f}%")

    plot_comparison(cfg, args.methods, args.betas)


if __name__ == "__main__":
    main()
