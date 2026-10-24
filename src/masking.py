"""
Masking policies for masked pre-training.

Every policy is the same sampler with a different per-patch weight: a patch
with a larger weight is more likely to be hidden. With all weights equal this
is exactly the random masking of the original MAE.

    fr : (B, N, 3) per-patch [background, cytoplasm, nucleus] fractions,
         from compartments.py
    returns keep (B, n_keep) and masked (B, N - n_keep) patch indices

Policies
    random   uniform weights (baseline)
    A        fixed compartment weights `pi`, e.g. hide cell patches 3x as often
    B        cross-compartment views: each image hides either ALL of its
             nucleus patches (reconstruct nucleus from cytoplasm) or ALL of its
             cytoplasm patches (reconstruct cytoplasm from nucleus)
    C        the same views as B, but the probability of each view is set by
             confusion feedback (see feedback.py) instead of being 50/50
"""

import torch

BG, CYTO, NUC = 0, 1, 2
HIDE_ALL = 1e4       # weight that makes a patch (almost) certain to be hidden


def sample(weights, mask_ratio=0.75):
    """Weighted sampling without replacement (Gumbel top-k).
    weights (B, N) > 0 -> keep, masked index tensors."""
    B, N = weights.shape
    n_keep = int(round(N * (1 - mask_ratio)))
    gumbel = -torch.log(-torch.log(torch.rand(B, N).clamp_(1e-9, 1 - 1e-9)))
    order = (weights.log() + gumbel).argsort(dim=1)      # low key = stays visible
    return order[:, :n_keep], order[:, n_keep:]


def weights_random(fr):
    return torch.ones(fr.shape[:2])


def weights_fixed(fr, pi):
    """Policy A. pi = (background, cytoplasm, nucleus) masking propensities."""
    return (fr * torch.tensor(pi, dtype=fr.dtype)).sum(-1).clamp_min(1e-3)


def weights_views(fr, view_probs):
    """Policies B and C. view_probs = (p_hide_cytoplasm, p_hide_nucleus).
    Each image draws one view; in that view every patch that is mostly the
    chosen compartment is hidden first, and the rest of the mask is random."""
    B = fr.size(0)
    hide_nucleus = torch.rand(B) < view_probs[1] / (view_probs[0] + view_probs[1])
    target = torch.where(hide_nucleus, NUC, CYTO)                       # (B,)
    major = fr.argmax(-1)                                               # (B, N)
    w = torch.ones(fr.shape[:2])
    w[major == target[:, None]] = HIDE_ALL
    return w


def make_policy(name, pi=(1.0, 3.0, 3.0)):
    """-> function(fr, state) -> weights. `state` carries view_probs for C."""
    if name == "random":
        return lambda fr, state: weights_random(fr)
    if name == "A":
        return lambda fr, state: weights_fixed(fr, pi)
    if name == "B":
        return lambda fr, state: weights_views(fr, (0.5, 0.5))
    if name == "C":
        return lambda fr, state: weights_views(fr, state["view_probs"])
    raise ValueError(name)
