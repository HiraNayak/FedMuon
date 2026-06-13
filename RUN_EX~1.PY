#!/usr/bin/env python3
"""Run a single federated optimization method and save its accuracy history.

Examples:
    python run_experiment.py --method fed_muon --beta 0.1
    python run_experiment.py --method scaffold --beta 10 --rounds 200   # quick test
"""

import argparse
import os

from src.config import CONFIG
from src.data import get_datasets
from src.methods import METHODS
from src.utils import get_device, load_json, save_json, set_seed
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("--beta", type=float, required=True, help="Dirichlet beta (0.1 or 10)")
    parser.add_argument("--rounds", type=int, default=None, help="Override CONFIG['rounds']")
    parser.add_argument("--seed", type=int, default=None, help="Override CONFIG['seeds'][0]")
    parser.add_argument("--out", default=None, help="Results JSON path (default: results/<method>.json)")
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

    history = METHODS[args.method](train_ds, test_loader, args.beta, cfg, seed, device)

    out_path = args.out or os.path.join(cfg["results_dir"], f"{args.method}.json")
    all_results = load_json(out_path)
    all_results[str(args.beta)] = history
    save_json(all_results, out_path)

    print(f"\nFinal accuracy:  {history[-1] * 100:.2f}%")
    print(f"Best accuracy:   {max(history) * 100:.2f}%")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
