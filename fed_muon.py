"""FedMuon: bias-corrected LMO-based federated optimization (Algorithm 1).

Each client corrects its momentum with (M_i - C_i + C) before applying the LMO,
where C_i is the client's own control variate from the previous round and C is
the server-side average of all clients' control variates. The server control
variate is updated ONCE per round from the accumulated deltas of sampled
clients (line 17) -- a common bug is to update C inside the per-client loop,
which double counts/staggers the correction.
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

    X = _get_params(model)
    C = _zeros_like(model)                       # server control variate
    C_i = [_zeros_like(model) for _ in range(n)]  # client control variates
    M_last = [_zeros_like(model) for _ in range(n)]
    history = []

    for r in range(R):
        sampled = random.sample(range(n), S)
        X_fin, C_new = {}, {}

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

                # Line 8: EMA momentum update
                M_i = [(1 - alpha) * m + alpha * g for m, g in zip(M_i, grad)]

                # Bias correction: M_i - C_i + C
                corrected = [m - ci + cv for m, ci, cv in zip(M_i, C_i[i], C)]

                # Line 9: parameter update via LMO on corrected momentum
                X_i = [xi + eta * di for xi, di in zip(X_i, lmo_apply(corrected, ns_steps))]

            # Line 11: C_i^{r+1} = M_i^{r,K}
            C_new[i] = [v.clone() for v in M_i]
            X_fin[i] = X_i
            M_last[i] = M_i

        # Line 17: C^{r+1} = C^r + (1/n) * sum_{i in S} (C_i^{r+1} - C_i^r)
        # Accumulate deltas for ALL sampled clients first, then update C once.
        delta_sum = _zeros_like(model)
        for i in sampled:
            delta_sum = _add(delta_sum, [cn - co for cn, co in zip(C_new[i], C_i[i])])
        C = _add(C, delta_sum, scale=1.0 / n)

        for i in sampled:
            C_i[i] = C_new[i]

        # Line 18: X^{r+1} = (n-S)/n * X^r + (1/n) * sum_{i in S} X_i^{r,K}
        X = _scale(X, (n - S) / n)
        for i in sampled:
            X = _add(X, X_fin[i], scale=1.0 / n)

        _set_params(model, X)
        acc = evaluate(model, test_loader, device)
        history.append(acc)
        if (r + 1) % 10 == 0 or r == 0:
            print(f"  [FedMuon      beta={beta}] round {r + 1:4d}/{R}  test_acc={acc * 100:.2f}%")

    del model
    free_memory()
    return history
