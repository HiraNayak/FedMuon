"""FedAvg with local Adam updates (Algorithm 1 aggregation, per-client Adam state)."""

import numpy as np
import torch
import torch.nn.functional as F

from ..data import CycleLoader, build_client_loaders
from ..models import LeNet
from ..utils import cpu_state_dict, evaluate, fedavg_aggregate, free_memory, load_state_dict


def run(train_ds, test_loader, beta, cfg, seed, device):
    n, S, K = cfg["n_clients"], cfg["s_clients"], cfg["local_steps"]
    lr, R = cfg["lr_adam"], cfg["rounds"]
    b1, b2, eps = 0.9, 0.999, 1e-8

    client_loaders = build_client_loaders(train_ds, n, beta, cfg["batch_size"], seed)

    model = LeNet().to(device)
    global_sd = cpu_state_dict(model)
    param_keys = [k for k, _ in model.named_parameters()]

    m1s = [{k: torch.zeros_like(global_sd[k], dtype=torch.float32) for k in param_keys} for _ in range(n)]
    m2s = [{k: torch.zeros_like(global_sd[k], dtype=torch.float32) for k in param_keys} for _ in range(n)]
    ts = [0] * n

    cyc = [CycleLoader(ldr) for ldr in client_loaders]
    history = []

    rng = np.random.default_rng(seed)
    for r in range(R):
        selected = rng.choice(n, S, replace=False)
        local_sds = []

        for i in selected:
            load_state_dict(model, global_sd, device)
            model.train()
            m1_i = {k: m1s[i][k].to(device) for k in param_keys}
            m2_i = {k: m2s[i][k].to(device) for k in param_keys}

            for _ in range(K):
                xb, yb = cyc[i].next()
                xb, yb = xb.to(device), yb.to(device)
                model.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                ts[i] += 1
                bc1 = 1 - b1 ** ts[i]
                bc2 = 1 - b2 ** ts[i]
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        m1_i[name].mul_(b1).add_(p.grad, alpha=1 - b1)
                        m2_i[name].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                        mh = m1_i[name] / bc1
                        vh = m2_i[name] / bc2
                        p.addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)

            m1s[i] = {k: m1_i[k].cpu() for k in param_keys}
            m2s[i] = {k: m2_i[k].cpu() for k in param_keys}
            local_sds.append(cpu_state_dict(model))
            free_memory()

        global_sd = fedavg_aggregate(global_sd, local_sds, n, S)
        del local_sds
        free_memory()

        load_state_dict(model, global_sd, device)
        acc = evaluate(model, test_loader, device)
        history.append(acc)
        print(f"  [FedAvg(Adam) beta={beta}] round {r + 1:4d}/{R}  test_acc={acc * 100:.2f}%")

    del model
    free_memory()
    return history
