# Federated Learning with Bias-Corrected LMO-Based Optimization 

A reproducibility study of **FedMuon: Federated Learning with Bias-Corrected LMO-Based
Optimization** ([Takezawa et al., arXiv:2509.26337](https://arxiv.org/abs/2509.26337)).

FedMuon adapts the [Muon optimizer](https://arxiv.org/abs/2502.16982) — a linear
minimization oracle (LMO) method that orthogonalizes momentum via Newton–Schulz
iteration — to the federated setting. Naively applying Muon in FL fails because the
LMO is a nonlinear operator: averaging each client's orthogonalized update is **not**
the same as orthogonalizing the averaged update. FedMuon corrects this bias with
client/server control variates before applying the LMO.

This repo reproduces the core FashionMNIST experiments from the paper and compares
FedMuon against four baselines.

## Methods implemented

| Method | File | Notes |
|---|---|---|
| FedAvg (SGD) | `src/methods/fedavg_sgd.py` | Local SGD + momentum |
| FedAvg (Adam) | `src/methods/fedavg_adam.py` | Per-client Adam state |
| SCAFFOLD | `src/methods/scaffold.py` | Variance reduction via control variates (Option II) |
| Local Muon | `src/methods/local_muon.py` | Muon locally, **no** bias correction (shows the bias problem) |
| FedMuon | `src/methods/fed_muon.py` | Muon + bias-corrected control variates (Algorithm 1) |

All methods share the same model (`src/models.py`), data partitioning
(`src/data.py`), evaluation/aggregation helpers (`src/utils.py`), and LMO
implementation (`src/lmo.py`, exact SVD by default, with a Newton–Schulz
approximation available for the ablation).

## Experimental setup

Matches Table 1 of the report (`FedMuon.pdf`):

| Hyperparameter | Value |
|---|---|
| Dataset | FashionMNIST (60k train / 10k test) |
| Model | LeNet-5 |
| Total clients (n) | 16 |
| Selected per round (S) | 8 |
| Local steps (K) | 5 |
| Batch size | 32 |
| Communication rounds | 1000 (2000 for the Newton–Schulz ablation) |
| Dirichlet beta | 0.1 (heterogeneous), 10.0 (homogeneous) |

Client data is split with a Dirichlet(beta) distribution over class labels
(`src/data.py:dirichlet_split`): beta=0.1 gives highly non-IID clients, beta=10
approximates IID.

## Usage

```bash
pip install -r requirements.txt

# Run one method/setting
python run_experiment.py --method fed_muon --beta 0.1

# Run everything (5 methods x 2 betas) and produce the comparison plot
python run_all.py

# Quick smoke test (few rounds, fast)
python run_all.py --rounds 20
```

Results are cached as JSON in `results/<method>.json` (keyed by beta) so runs can
be resumed. `run_all.py` also writes `results/comparison.png`.

## Expected results 

- **Heterogeneous (beta=0.1):** FedMuon reaches ~89-90% test accuracy after 1000
  rounds, ahead of FedAvg(SGD)/SCAFFOLD(Adam) (~85-87%). **Local Muon stalls at
  ~65-68%**, confirming that the LMO bias collapses accuracy without correction.
- **Homogeneous (beta=10):** FedMuon and Local Muon converge to similar accuracy
  (~90%), since local gradients are already aligned and the correction term is small.
- **Newton-Schulz ablation:** accuracy improves monotonically with iteration count
  T in {0, 1, 2, 4}; the gap between T=2 and T=4 is small relative to T=0 vs T=1.

## Repo structure

```
fedmuon-repro/
├── FedMuon.pdf          # reproducibility report
├── run_experiment.py    # run a single method/beta
├── run_all.py           # run everything + plot
├── requirements.txt
├── src/
│   ├── config.py        # shared hyperparameters
│   ├── models.py         # LeNet-5
│   ├── data.py           # FashionMNIST + Dirichlet partitioning
│   ├── lmo.py             # exact / Newton-Schulz LMO
│   ├── plot.py            # comparison plots
│   ├── utils.py           # seeding, evaluation, FedAvg aggregation, JSON I/O
│   └── methods/
│       ├── fedavg_sgd.py
│       ├── fedavg_adam.py
│       ├── scaffold.py
│       ├── local_muon.py
│       └── fed_muon.py
└── results/              # JSON histories + plots (generated)
```

## Authors

Hira Nayak, Samprithi Suresh Premalatha, Krisha Chemburkar

## References

1. Takezawa et al. *FedMuon: Federated Learning with Bias-corrected LMO-based
   Optimization*. [arXiv:2509.26337](https://arxiv.org/abs/2509.26337)
2. McMahan et al. *Communication-Efficient Learning of Deep Networks from
   Decentralized Data* (FedAvg).
3. Karimireddy et al. *SCAFFOLD: Stochastic Controlled Averaging for Federated
   Learning*.
4. Jordan et al. *Muon: An optimizer for hidden layers in neural networks*.
   [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
