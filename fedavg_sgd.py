"""FedAvg with local SGD + momentum (Algorithm 1 aggregation, plain SGD local steps)."""

import numpy as np
import torch
import torch.nn.functional as F

from ..data import CycleLoader, build_client_loaders
from ..models import LeNet
from ..utils import cpu_state_dict, evaluate, fedavg_aggregate, free_memory, load_state_dict


def run(train_ds, test_loader, beta, cfg, seed, device):
    n, S, K = cfg["n_clients"], cfg["s_clients"], cfg["local_steps"]
    lr, mom, R = cfg["lr_sgd"], cfg["momentum"], cfg["rounds"]

    client_loaders = build_client_loaders(train_ds, n, beta, cfg["batch_size"], seed)

    model = LeNet().to(device)
    global_sd = cpu_state_dict(model)
    param_keys = [k for k, _ in model.named_parameters()]

    mom_buf = [{k: torch.zeros_like(global_sd[k], dtype=torch.float32) for k in param_keys}
               for _ in range(n)]

    cyc = [CycleLoader(ldr) for ldr in client_loaders]
    history = []

    rng = np.random.default_rng(seed)
    for r in range(R):
        selected = rng.choice(n, S, replace=False)
        local_sds = []

        for i in selected:
            load_state_dict(model, global_sd, device)
            model.train()
            buf_i = {k: mom_buf[i][k].to(device) for k in param_keys}

            for _ in range(K):
                xb, yb = cyc[i].next()
                xb, yb = xb.to(device), yb.to(device)
                model.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        buf_i[name].mul_(mom).add_(p.grad)
                        p.add_(buf_i[name], alpha=-lr)

            mom_buf[i] = {k: buf_i[k].cpu() for k in param_keys}
            local_sds.append(cpu_state_dict(model))
            free_memory()

        global_sd = fedavg_aggregate(global_sd, local_sds, n, S)
        del local_sds
        free_memory()

        load_state_dict(model, global_sd, device)
        acc = evaluate(model, test_loader, device)
        history.append(acc)
        print(f"  [FedAvg(SGD)  beta={beta}] round {r + 1:4d}/{R}  test_acc={acc * 100:.2f}%")

    del model
    free_memory()
    return history
