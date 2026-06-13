"""Shared helpers: reproducibility, device/memory, evaluation, FedAvg aggregation, JSON I/O."""

import gc
import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict(model, sd, device):
    model.load_state_dict({k: v.to(device) for k, v in sd.items()})


@torch.no_grad()
def evaluate(model, loader, device):
    """Return classification accuracy of `model` on `loader`."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def fedavg_aggregate(global_sd, local_sds, n, s):
    """Algorithm 1, line 18: X^{r+1} = (n-S)/n * X^r + (1/n) * sum_{i in S} X_i^{r,K}."""
    out = {}
    for k in global_sd:
        local_sum = torch.stack([sd[k].float() for sd in local_sds]).sum(0)
        out[k] = (n - s) / n * global_sd[k].float() + local_sum / n
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
