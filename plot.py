"""Plot test-accuracy comparisons across methods (Figure 1 style)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import load_json

LABELS = {
    "fedavg_sgd": "FedAvg (SGD)",
    "fedavg_adam": "FedAvg (Adam)",
    "scaffold": "SCAFFOLD",
    "local_muon": "Local Muon",
    "fed_muon": "FedMuon",
}

STYLE = {
    "fedavg_sgd": dict(color="#4C72B0", linestyle="--"),
    "fedavg_adam": dict(color="#DD8452", linestyle="--"),
    "scaffold": dict(color="#55A868", linestyle="-."),
    "local_muon": dict(color="tab:purple", linestyle="-"),
    "fed_muon": dict(color="tab:brown", linestyle="-", linewidth=2.4),
}


def plot_comparison(cfg, methods, betas, out_path=None):
    out_path = out_path or os.path.join(cfg["results_dir"], "comparison.png")

    fig, axes = plt.subplots(1, len(betas), figsize=(6 * len(betas), 4.5), squeeze=False)
    axes = axes[0]

    for ax, beta in zip(axes, betas):
        for method in methods:
            results = load_json(os.path.join(cfg["results_dir"], f"{method}.json"))
            hist = results.get(str(beta))
            if not hist:
                continue
            ax.plot(range(1, len(hist) + 1), [v * 100 for v in hist],
                    label=LABELS.get(method, method), **STYLE.get(method, {}))

        regime = "Heterogeneous" if beta < 1 else "Homogeneous"
        ax.set_title(f"FashionMNIST -- {regime} (beta={beta})")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_ylim(40, 100)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")
