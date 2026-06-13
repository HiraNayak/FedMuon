from . import fedavg_sgd, fedavg_adam, local_muon, fed_muon, scaffold

METHODS = {
    "fedavg_sgd": fedavg_sgd.run,
    "fedavg_adam": fedavg_adam.run,
    "local_muon": local_muon.run,
    "fed_muon": fed_muon.run,
    "scaffold": scaffold.run,
}
