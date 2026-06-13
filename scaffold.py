"""SCAFFOLD (Karimireddy et al.) with local SGD -- variance-reduction baseline.

Rewritten from an initial draft to match the shared experimental setup used by
the other methods (n=16, S=8, K local mini-batch steps, batch_size, beta in
{10, 0.1}, FedAvg-style weighted aggregation -- Algorithm 1 line 18) instead of
the original draft's "local_epochs over the whole client dataset" + unweighted
mean aggregation.

Per-step correction (Karimireddy et al., Algorithm 1):
    grad_i <- grad_i - c_i + c
Client control variate update (Option II):
    c_i^{new} = c_i - c + (x_global - x_i_local) / (K * lr)
Server control variate update:
    c <- c + (1/n) * sum_{i in S} (c_i^{new} - c_i)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import CycleLoader, build_client_loaders
from ..models import LeNet
from ..utils import cpu_state_dict, evaluate, fedavg_aggregate, free_memory, load_state_dict


def run(train_ds, test_loader, beta, cfg, seed, device):
    n, S, K, R = cfg["n_clients"], cfg["s_clients"], cfg["local_steps"], cfg["rounds"]
    lr = cfg["scaffold_lr"]

    client_loaders = build_client_loaders(train_ds, n, beta, cfg["batch_size"], seed)

    model = LeNet().to(device)
    global_sd = cpu_state_dict(model)
    param_keys = [k for k, _ in model.named_parameters()]

    server_control = {k: torch.zeros_like(global_sd[k], dtype=torch.float32) for k in param_keys}
    client_control = [{k: torch.zeros_like(global_sd[k], dtype=torch.float32) for k in param_keys}
                       for _ in range(n)]

    cyc = [CycleLoader(ldr) for ldr in client_loaders]
    history = []

    rng = np.random.default_rng(seed)
    for r in range(R):
        selected = rng.choice(n, S, replace=False)
        local_sds = []
        new_controls = {}

        for i in selected:
            load_state_dict(model, global_sd, device)
            model.train()

            c_i = {k: client_control[i][k].to(device) for k in param_keys}
            c = {k: server_control[k].to(device) for k in param_keys}

            for _ in range(K):
                xb, yb = cyc[i].next()
                xb, yb = xb.to(device), yb.to(device)
                model.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        corrected_grad = p.grad - c_i[name] + c[name]
                        p.add_(corrected_grad, alpha=-lr)

            # Option II client control update:
            # c_i_new = c_i - c + (x_global - x_local) / (K * lr)
            new_c_i = {}
            with torch.no_grad():
                for name, p in model.named_parameters():
                    global_p = global_sd[name].to(device)
                    new_c_i[name] = (c_i[name] - c[name]
                                      + (global_p - p.data) / (K * lr))
            new_controls[i] = {k: v.cpu() for k, v in new_c_i.items()}

            local_sds.append(cpu_state_dict(model))
            free_memory()

        # Aggregate model weights (Algorithm 1, line 18)
        global_sd = fedavg_aggregate(global_sd, local_sds, n, S)

        # Server control update: c <- c + (1/n) * sum_{i in S} (c_i_new - c_i_old)
        delta_sum = {k: torch.zeros_like(server_control[k]) for k in param_keys}
        for i in selected:
            for k in param_keys:
                delta_sum[k] += new_controls[i][k] - client_control[i][k]
        for k in param_keys:
            server_control[k] += delta_sum[k] / n

        for i in selected:
            client_control[i] = new_controls[i]

        del local_sds
        free_memory()

        load_state_dict(model, global_sd, device)
        acc = evaluate(model, test_loader, device)
        history.append(acc)
        print(f"  [SCAFFOLD     beta={beta}] round {r + 1:4d}/{R}  test_acc={acc * 100:.2f}%")

    del model
    free_memory()
    return history
