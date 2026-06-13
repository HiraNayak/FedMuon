"""LocalMuon: vanilla FedAvg with Muon (EMA momentum + LMO) as the local optimizer.

No bias correction -- each client independently orthogonalizes its own momentum
before the server averages the resulting updates. Included as the baseline that
demonstrates the bias problem described in Section 3.2 of the report.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import build_client_loaders
from ..lmo import lmo_apply
from ..models import LeNet
from ..utils import evaluate, free_memory


def _get_params(model):
    return [p.data.clone() for p in model.parameters()]


def _set_params(model, params):
    for dst, src in zip(model.parameters(), params):
        dst.data.copy_(src)


def _zeros_like(model):
    return [torch.zeros_like(p.data) for p in model.parameters()]


def _add(a, b, scale=1.0):
    return [x + scale * y for x, y in zip(a, b)]


def _scale(a, scale):
    return [scale * x for x in a]


def _next_batch(loader, it, device):
    try:
        xb, yb = next(it)
    except StopIteration:
        it = iter(loader)
        xb, yb = next(it)
    return xb.to(device), yb.to(device), it


def run(train_ds, test_loader, beta, cfg, seed, device):
    n, S, K, R = cfg["n_clients"], cfg["s_clients"], cfg["local_steps"], cfg["rounds"]
    eta, alpha = cfg["muon_eta"], cfg["muon_alpha"]
    ns_steps = cfg["newton_schulz_T"]

    loaders = build_client_loaders(train_ds, n, beta, cfg["batch_size"], seed)
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()

    X = _get_params(model)
    M_last = [_zeros_like(model) for _ in range(n)]
    history = []

    for r in range(R):
        sampled = random.sample(range(n), S)
        X_fin = {}

        for i in sampled:
            X_i = [v.clone() for v in X]
            M_i = [v.clone() for v in M_last[i]]
            it = iter(loaders[i])

            for _ in range(K):
                xb, yb, it = _next_batch(loaders[i], it, device)
                _set_params(model, X_i)
                model.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                grad = [p.grad.data.clone() for p in model.parameters()]

                M_i = [(1 - alpha) * m + alpha * g for m, g in zip(M_i, grad)]
                X_i = [xi + eta * di for xi, di in zip(X_i, lmo_apply(M_i, ns_steps))]

            X_fin[i] = X_i
            M_last[i] = M_i

        X = _scale(X, (n - S) / n)
        for i in sampled:
            X = _add(X, X_fin[i], scale=1.0 / n)

        _set_params(model, X)
        acc = evaluate(model, test_loader, device)
        history.append(acc)
        if (r + 1) % 10 == 0 or r == 0:
            print(f"  [LocalMuon    beta={beta}] round {r + 1:4d}/{R}  test_acc={acc * 100:.2f}%")

    del model
    free_memory()
    return history
