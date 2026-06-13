"""Default experiment configuration, matching Table 1 of the FedMuon report."""

CONFIG = {
    "dataset": "FashionMNIST",
    "data_dir": "./data",
    "results_dir": "./results",

    # Federated setup
    "n_clients": 16,
    "s_clients": 8,
    "local_steps": 5,
    "batch_size": 32,
    "rounds": 1000,

    # Data heterogeneity: 0.1 = heterogeneous, 10.0 = homogeneous
    "beta_vals": [10.0, 0.1],
    "seeds": [42],

    # FedAvg baselines
    "lr_sgd": 0.01,
    "momentum": 0.9,
    "lr_adam": 1e-3,

    # Muon-based methods (LocalMuon, FedMuon)
    "muon_eta": 0.01,
    "muon_alpha": 0.9,
    "newton_schulz_T": None,  # None => exact SVD-based LMO

    # SCAFFOLD
    "scaffold_lr": 0.1,
}
