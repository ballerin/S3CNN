"""Assembled models.

These are contract tests, not accuracy tests: shapes, purity, gradients,
equivariance of the whole stack, and a set of golden outputs. The goldens are
pinned against the code as shipped, so any milestone that changes the numbers
has to say so out loud.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.SO3CNN import SO3CNN
from tests import models
from tests.conftest import N_ALPHA, N_BETA, relerr, tables

GOLDENS = "tests/goldens.npz"
ROLLS = [3, 6, 9]


@pytest.fixture(scope="module")
def x():
    return models.sample_input()


# --------------------------------------------------------------- contracts

@pytest.mark.parametrize("name", models.GRID_MODELS)
def test_grid_model_returns_the_input_grid(name, x):
    y = models.build(name)(x)
    assert y.shape == (x.size(0), N_ALPHA, N_BETA)
    assert y.dtype == torch.complex64
    assert torch.isfinite(y.real).all() and torch.isfinite(y.imag).all()


@pytest.mark.parametrize("name", ["mnist_spins", "mnist_so3"])
def test_classifier_returns_real_logits(name, x):
    y = models.build(name)(x)
    assert y.shape == (x.size(0), 10)
    assert y.dtype == torch.float32 and torch.isfinite(y).all()


@pytest.mark.parametrize("name", list(models.BUILDERS))
def test_samples_do_not_interact(name, x):
    """No batch norm in eval, so the batch dimension must be independent."""
    m = models.build(name)
    assert relerr(m(x)[:1].detach().numpy(), m(x[:1]).detach().numpy()) < 1e-6


@pytest.mark.parametrize("name", list(models.BUILDERS))
def test_gradients_are_finite(name, x):
    m = models.build(name)
    m(x).abs().sum().backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradient for {bad}"


# ------------------------------------------------------------ equivariance

@pytest.mark.parametrize("name", models.GRID_MODELS)
@pytest.mark.parametrize("k", ROLLS)
def test_z_rotation_equivariance(name, k, x):
    """Rolling the longitude grid must roll the output by the same amount.

    Exact up to float32 because every layer here shares a longitude grid, or
    pools to one that divides it. `test_pooling_needs_commensurate_grids`
    covers the case where it does not.
    """
    m = models.build(name)
    got = m(torch.roll(x, k, dims=1)).detach().numpy()
    want = torch.roll(m(x).detach(), k, dims=1).numpy()
    assert relerr(got, want) < 1e-5


@pytest.mark.parametrize("name", ["mnist_spins", "mnist_so3"])
@pytest.mark.parametrize("k", ROLLS)
def test_classifier_is_z_rotation_invariant(name, k, x):
    """The head truncates to L = 0, which is the invariant subspace."""
    m = models.build(name)
    got = m(torch.roll(x, k, dims=1)).detach().numpy()
    assert relerr(got, m(x).detach().numpy()) < 1e-5


@pytest.mark.parametrize("k", [3, 6])
def test_pooling_needs_commensurate_grids(k, x):
    """Pooling to L' puts the nonlinearity on a grid of 2(L'+1) longitudes.

    A pointwise nonlinearity commutes with a rotation only when that rotation
    is a whole number of grid steps, so equivariance survives pooling only if
    the inner grid divides the outer one. 2(4+1) = 10 does not divide
    N_ALPHA = 18, and the error grows by four orders of magnitude. Only rolls
    that are not a whole step on both grids are used: 9 is half a turn, which
    is exact everywhere.
    """
    torch.manual_seed(models.SEED)
    coarse = models.unet("spins", inner=4).eval()
    got = coarse(torch.roll(x, k, dims=1)).detach().numpy()
    want = torch.roll(coarse(x).detach(), k, dims=1).numpy()
    assert relerr(got, want) > 1e-5


# ---------------------------------------------------- known-defect pinning

@pytest.mark.parametrize("style", ["so3", "full_so3"])
def test_batch_norm_works_with_so3_layers(style, x):
    """The SO(3) styles hand the normalisation a list of Wigner blocks, so they
    need `FullSpectralBatchNormalization`, not the flat-layout one."""
    n0, n1, full = tables(models.L, N_BETA)
    out = SO3CNN(n_alpha=N_ALPHA, n_beta=N_BETA, layer_style=style, resolution=models.L,
                 widths=[1, 2, 1], spins=[[0], [0, 1], [0]], use_batch_norm=True,
                 nonlinearity="leaky_relu", wigner_n0=n0, wigner_n1=n1,
                 f_wigner=full).eval()(x)
    assert torch.isfinite(torch.view_as_real(out)).all()


def test_spins_may_be_bare_ints():
    """The documented `[0, [0,1], 0]` form: a bare int is one spin."""
    n0, n1, full = tables(models.L, N_BETA)
    SO3CNN(n_alpha=N_ALPHA, n_beta=N_BETA, layer_style="spins", resolution=models.L,
           widths=[1, 2, 1], spins=[0, [0, 1], 0], use_batch_norm=False,
           wigner_n0=n0, wigner_n1=n1, f_wigner=full)


def test_no_parameter_is_dead(x):
    """Modules a `forward` never reaches must not be built: their parameters
    would reach the optimizer and the state_dict, take weight decay, and never
    receive a gradient."""
    dead = {}
    for name in models.BUILDERS:
        m = models.build(name)
        m(x).abs().sum().backward()
        got = [n for n, p in m.named_parameters() if p.grad is None]
        if got:
            dead[name] = got
    assert not dead, dead


# ----------------------------------------------------------------- goldens

@pytest.mark.parametrize("name", list(models.BUILDERS))
def test_output_matches_golden(name, x, request):
    """Pinned against the code as shipped, at seed `models.SEED`.

    Kernels are currently constant (defect 1), so these do not yet depend on
    the seed. Fixing that changes every golden here, on purpose: regenerate
    with `pytest --update-goldens` and say so in the commit.
    """
    if request.config.getoption("--update-goldens"):
        pytest.skip("regenerating")
    with np.load(GOLDENS) as f:
        assert relerr(models.build(name)(x).detach().numpy(), f[name]) < 1e-5


def test_regenerate_goldens(x, request):
    if not request.config.getoption("--update-goldens"):
        pytest.skip("pass --update-goldens to rewrite")
    np.savez_compressed(GOLDENS, **{n: models.build(n)(x).detach().numpy()
                                    for n in models.BUILDERS})
