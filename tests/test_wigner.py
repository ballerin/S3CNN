"""Wigner d-matrices: agreement with an independent formula, and the
algebraic identities the transforms rely on."""

from __future__ import annotations

import numpy as np
import pytest
from modules.SO3lib import bDelta, vectorized_wigner_d_matrix, wigner_d_matrix
from tests import reference as ref

BETAS = np.array([0.05, 0.7, 1.9, 3.1])
ORDERS = [0, 1, 2, 5, 10, 16]


@pytest.mark.parametrize("l", ORDERS)
def test_matches_factorial_formula(l):
    """`modules.SO3lib` uses a matrix exponential; the reference does not."""
    got = vectorized_wigner_d_matrix(l, BETAS).numpy()
    assert np.abs(got - ref.wigner_d(l, BETAS)).max() < 1e-6


@pytest.mark.parametrize("l", ORDERS)
def test_real(l):
    """d^l is real, yet it is stored and multiplied as complex64."""
    assert vectorized_wigner_d_matrix(l, BETAS).imag.abs().max() == 0


@pytest.mark.parametrize("l", ORDERS)
def test_orthogonal_and_representation(l):
    d = vectorized_wigner_d_matrix(l, np.array([0.3, 0.4, 0.7])).numpy().real
    eye = np.eye(2 * l + 1)
    assert np.abs(d[0] @ d[0].T - eye).max() < 1e-5
    assert np.abs(d[0] @ d[1] - d[2]).max() < 1e-5          # d(a)d(b) = d(a+b)
    assert np.abs(d[0].T - vectorized_wigner_d_matrix(l, np.array([-0.3]))
                  .numpy().real[0]).max() < 1e-5            # d(-b) = d(b)^T


@pytest.mark.parametrize("l", [1, 3, 5])
def test_scalar_and_vectorised_agree(l):
    got = vectorized_wigner_d_matrix(l, np.array([0.7])).numpy()[0]
    assert np.abs(got - wigner_d_matrix(l, 0.7).numpy()).max() < 1e-6


@pytest.mark.parametrize("l", [1, 2, 5, 10])
def test_delta_decomposition(l):
    """d^l_{m,n}(b) = i^{n-m} sum_s D^l_{s,m} D^l_{s,n} exp(i s b).

    Paper appendix B; the identity the unimplemented "delta" FFT rests on.
    """
    delta = bDelta(l, vectorized_wigner_d_matrix(l, np.array([np.pi / 2]))[0])
    delta = delta.numpy() / (-1j) ** np.arange(-l, l + 1)[:, None]
    m = np.arange(-l, l + 1)
    phase = (1j) ** (m[None, :] - m[:, None])
    for beta in (0.4, 2.2):
        got = np.einsum("sm,sn,s->mn", delta, delta, np.exp(1j * m * beta)) * phase
        assert np.abs(got - ref.wigner_d(l, np.array([beta]))[0]).max() < 1e-4
