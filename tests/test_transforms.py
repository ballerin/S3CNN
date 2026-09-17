"""S2 and SO(3) transforms.

Synthesis is exact by definition, so it is checked against a direct sum.
Analysis involves a quadrature, so it is checked against the ideal L2
projection.  Tests marked `xfail(strict=True)` pin a known defect: fixing one
turns it red, which is the point.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.SO3fft import S2FFT, S2iFFT, SO3FFT, SO3iFFT
from modules.SpectralModules import SpatialSmoothing
from tests import reference as ref
from tests.conftest import L, N_ALPHA, N_BETA, random_spectrum, relerr, tables

SPINS = [0, 1]


def _synth(f_hat, n, wigner, n_beta, n_alpha=N_ALPHA):
    n0, n1, _ = wigner
    return S2iFFT(L=L, n=n, n_alpha=n_alpha, n_beta=n_beta)(
        f_hat, d_matrices_n0=n0, d_matrices_n1=n1)


def _analyse(f, n, wigner):
    n0, n1, _ = wigner
    return S2FFT(L=L, n=n)(f, d_matrices_n0=n0, d_matrices_n1=n1)


def _roundtrip_error(n, n_beta):
    w = tables(L, n_beta)
    f_hat = random_spectrum(L, n)
    return relerr(_analyse(_synth(f_hat, n, w, n_beta), n, w).numpy(), f_hat.numpy())


# --------------------------------------------------------------- synthesis

@pytest.mark.parametrize("n", SPINS)
def test_synthesis_matches_direct_sum(n, wigner, betas):
    """Synthesis carries no quadrature, so it should be exact to complex64."""
    f_hat = random_spectrum(L, n)
    got = _synth(f_hat, n, wigner, N_BETA).numpy()[0, 0]
    want = ref.synthesize_s2(f_hat.numpy()[0, 0], n, N_ALPHA, betas)
    assert relerr(got, want) < 1e-5


# ---------------------------------------------------------------- analysis

@pytest.mark.parametrize("n", SPINS)
def test_analysis_matches_ideal_projection(n):
    """Against the exact L2 projection, at eight times the minimum sampling."""
    n_beta = 8 * (L + 1)
    w = tables(L, n_beta)
    f_hat = random_spectrum(L, n).numpy()[0, 0]
    f = _synth(torch.tensor(f_hat)[None, None], n, w, n_beta)
    want = ref.analyze_s2(lambda a, b: ref.synthesize_s2(f_hat, n, a.size, b),
                          n, L, N_ALPHA)
    assert relerr(_analyse(f, n, w).numpy(), want) < 1e-2


@pytest.mark.parametrize("n", SPINS)
@pytest.mark.parametrize("n_beta,tol", [(2 * (L + 1), 8e-2), (4 * (L + 1), 2e-2),
                                        (8 * (L + 1), 5e-3)])
def test_roundtrip(n, n_beta, tol):
    """Current accuracy of analysis(synthesis(x)) == x, as shipped."""
    assert _roundtrip_error(n, n_beta) < tol


@pytest.mark.xfail(strict=True, reason="defect 2: quadrature is only O(n_beta^-2)")
def test_roundtrip_is_spectrally_accurate():
    """A correct quadrature is exact on band-limited data at any oversampling.

    The shipped one converges as O(n_beta^-2) instead, so reaching single
    precision would need thousands of latitudes.  Goes green when M6 lands.
    """
    assert _roundtrip_error(0, 4 * (L + 1)) < 1e-6


def test_roundtrip_convergence_is_second_order():
    """Doubling the latitudes divides the error by four, not by much more."""
    ratio = _roundtrip_error(0, 4 * (L + 1)) / _roundtrip_error(0, 8 * (L + 1))
    assert 3.5 < ratio < 4.5


@pytest.mark.xfail(strict=True, reason="defect 2: L = n_beta is rank-deficient")
@pytest.mark.parametrize("n", SPINS)
def test_roundtrip_at_production_band_limit(n):
    """Every shipped ERA5 and MNIST model sets L = n_beta."""
    assert _roundtrip_error(n, L) < 1e-2


@pytest.mark.parametrize("n_beta,tol", [(L, 0.8), (2 * (L + 1), 5e-2),
                                        (8 * (L + 1), 3e-3)])
def test_discrete_orthogonality(n_beta, tol):
    """The quadrature should give <d^l_{0,0}, d^l_{0,0}> = 1 at every order.

    At L = n_beta the top order keeps a third of its norm; the network's
    highest frequencies are attenuated on every transform.
    """
    _, bs = ref.grid(1, n_beta)
    col = ref.wigner_column(L, bs, 0)[:, L]
    norm = (2 * L + 1) / 2 * np.pi / (n_beta + 1) * np.sum(np.sin(bs) * col ** 2)
    assert abs(norm - 1) < tol


# ------------------------------------------------------------------- SO(3)

def _so3_roundtrip_error(l_max, n_beta):
    _, _, full = tables(l_max, n_beta)
    n_ang = 2 * (l_max + 1)
    x_hat = [torch.randn(1, 2, 2 * l + 1, 2 * l + 1, dtype=torch.complex64)
             for l in range(l_max + 1)]
    x = SO3iFFT(L=l_max, n_alpha=n_ang, n_beta=n_beta, n_gamma=n_ang)(
        x_hat, d_matrices=full)
    back = SO3FFT(L=l_max)(x, d_matrices=full)
    return max(relerr(back[l].numpy(), x_hat[l].numpy()) for l in range(l_max + 1))


@pytest.mark.parametrize("n_beta", [40, 80, 179])
def test_so3_roundtrip(n_beta):
    """SO3FFT used to read its quadrature nodes at a hard-coded 179 latitudes."""
    assert _so3_roundtrip_error(6, n_beta) < 5e-2


# --------------------------------------------------------------- smoothing

@pytest.mark.parametrize("n", SPINS)
def test_spatial_smoothing_selects_the_spectral_column(n, wigner):
    """S_q on the grid agrees with column q of each Wigner block, up to scale."""
    _, _, full = wigner
    n_gamma = 2 * (L + 1)
    x_hat = [torch.randn(1, 2, 2 * l + 1, 2 * l + 1, dtype=torch.complex64)
             for l in range(L + 1)]
    x = SO3iFFT(L=L, n_alpha=n_gamma, n_beta=N_BETA, n_gamma=n_gamma)(
        x_hat, d_matrices=full)
    got = _analyse(SpatialSmoothing(n=n)(x), n, wigner)

    want = torch.zeros_like(got)
    for l in range(abs(n), L + 1):
        want[..., l * l:(l + 1) * (l + 1)] = x_hat[l][..., l + n]
    assert relerr(got.numpy(), want.numpy()) < 8e-2


def test_so3_and_s2_paths_share_a_normalisation():
    """Going SO(3) -> grid -> smoothing -> S2 spectrum must return the column.

    SO3iFFT leaves gamma unnormalised precisely so that SpatialSmoothing's
    1/n_gamma average inverts it; alpha follows the S2 convention. A 2-D
    `norm="ortho"` on both axes would leave a 1/sqrt(n_gamma) that varies with
    the resolution schedule, so the factor would differ from layer to layer.
    """
    w = tables(6, 40)
    n_gamma = 14
    x_hat = [torch.randn(1, 1, 2 * l + 1, 2 * l + 1, dtype=torch.complex64)
             for l in range(7)]
    x = SO3iFFT(L=6, n_alpha=n_gamma, n_beta=40, n_gamma=n_gamma)(
        x_hat, d_matrices=w[2])
    got = S2FFT(L=6, n=0)(SpatialSmoothing(n=0)(x),
                          d_matrices_n0=w[0], d_matrices_n1=w[1])
    want = torch.zeros_like(got)
    for l in range(7):
        want[..., l * l:(l + 1) * (l + 1)] = x_hat[l][..., l]
    assert relerr(got.numpy(), want.numpy()) < 5e-2
