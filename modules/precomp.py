"""Loading the precomputed Wigner tables from `precomp/`.

The tables depend on both the latitude count and the maximum order, so the
filename carries both. Files written before that convention are named by
`n_beta` alone and are still accepted, since regenerating them takes hours.
"""

from __future__ import annotations

import os
import pickle

DIR = "precomp"


def path(kind: str, n_beta: int, l_max: int | None = None) -> str:
    """The file to read, preferring the two-part name when it exists."""
    if l_max is not None:
        both = os.path.join(DIR, f"{kind}_{n_beta}_{l_max}.list")
        if os.path.exists(both):
            return both
    return os.path.join(DIR, f"{kind}_{n_beta}.list")


def load(kind: str, n_beta: int, l_max: int | None = None) -> list:
    """A table as a list of tensors, detached from the autograd graph."""
    with open(path(kind, n_beta, l_max), "rb") as fh:
        return [m.requires_grad_(False) for m in pickle.load(fh)]
