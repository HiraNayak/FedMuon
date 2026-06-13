"""Linear minimization oracle (LMO) used by LocalMuon and FedMuon.

lmo(G) = -U V^T, where G = U Sigma V^T (the polar factor of G), computed via
exact SVD. Vectors (e.g. bias terms) are normalized to unit norm instead.

A Newton-Schulz iterative approximation is also provided for the ablation
described in the report (varying iteration count T).
"""

import torch


def exact_lmo(G: torch.Tensor) -> torch.Tensor:
    if G.dim() == 1:
        norm = G.norm()
        return -(G / norm) if norm > 1e-12 else torch.zeros_like(G)

    orig_shape = G.shape
    G2 = G.view(G.shape[0], -1) if G.dim() > 2 else G
    try:
        U, _, Vh = torch.linalg.svd(G2, full_matrices=False)
        return -(U @ Vh).view(orig_shape)
    except RuntimeError:
        fro = G2.norm(p="fro")
        out = -(G2 / fro) if fro > 1e-12 else torch.zeros_like(G2)
        return out.view(orig_shape)


def newton_schulz_lmo(G: torch.Tensor, steps: int = 4) -> torch.Tensor:
    """Quadratically-convergent polynomial approximation to exact_lmo.

    steps=0 returns the (normalized) input unchanged, i.e. no orthogonalization.
    """
    if G.dim() == 1:
        norm = G.norm()
        return -(G / norm) if norm > 1e-12 else torch.zeros_like(G)

    orig_shape = G.shape
    X = G.view(G.shape[0], -1) if G.dim() > 2 else G
    X = X / (X.norm(p="fro") + 1e-12)

    if steps <= 0:
        return -X.view(orig_shape)

    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transpose:
        X = X.T

    return -X.view(orig_shape)


def lmo_apply(params, steps=None):
    """Apply LMO to each tensor in `params`.

    steps=None -> exact SVD-based LMO. steps=int -> Newton-Schulz with that
    many iterations (steps=0 means no orthogonalization, just normalization).
    """
    if steps is None:
        return [exact_lmo(p) for p in params]
    return [newton_schulz_lmo(p, steps) for p in params]
