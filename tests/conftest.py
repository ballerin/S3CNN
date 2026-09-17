"""Shared fixtures.

Wigner tables are built from `tests.reference` rather than read from
`precomp/`, so the suite runs on a fresh clone.  Sizes are small on purpose:
every test here is a correctness test, not a benchmark.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch

from tests import reference as ref

L = 8
N_BETA = 20
N_ALPHA = 2 * (L + 1)


def _tables(l_max: int, n_beta: int):
    _, betas = ref.grid(1, n_beta)
    full = [torch.tensor(ref.wigner_d(l, betas), dtype=torch.complex64)
            for l in range(l_max + 1)]
    n0 = [t[:, :, l] for l, t in enumerate(full)]
    n1 = [t[:, :, l + 1] for l, t in enumerate(full) if l >= 1]
    return [t.contiguous() for t in n0], [t.contiguous() for t in n1], full


tables = functools.lru_cache(maxsize=8)(_tables)


@pytest.fixture(scope="session")
def wigner():
    """(spin-0 columns, spin-1 columns, full matrices) at the default size."""
    return tables(L, N_BETA)


@pytest.fixture(scope="session")
def betas():
    return ref.grid(1, N_BETA)[1]


def random_spectrum(l_max: int, n: int, channels: int = 1, batch: int = 1,
                    seed: int = 0) -> torch.Tensor:
    """A band-limited spectrum in the flat layout, with the spin-1 zero set."""
    rng = np.random.default_rng(seed)
    shape = (batch, channels, (l_max + 1) ** 2)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    if n == 1:
        x[..., 0] = 0
    return torch.tensor(x, dtype=torch.complex64)


def relerr(got, want) -> float:
    """Max relative error; 0 when both sides are structurally zero."""
    got, want = np.asarray(got), np.asarray(want)
    scale = np.abs(want).max()
    return float(np.abs(got - want).max() / scale) if scale else float(np.abs(got).max())


def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true",
                     help="rewrite tests/goldens.npz from the current code")
