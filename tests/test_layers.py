"""Individual layers: equivariance of the convolutions, purity of every
module, and unit tests pinning the arithmetic defects in the normalisations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.SO3ConvLayer import SO3Conv, SO3toSO3Conv, SpinSO3Conv
from modules.SpectralModules import (ComplexDropout, FullSpectralBatchNormalization,
                                     FullSpectralLayerNormalization,
                                     MagnitudeNonlinearity,
                                     PhaseCollapseNonlinearity, RealNonlinearity,
                                     SphericalBatchNormalization,
                                     SpinSpectralBatchNormalization,
                                     SpectralDropout, SpectralSmoothing,
                                     SpinSpectralLayerNormalization,
                                     WeightedLeakyReLU)
from tests import reference as ref
from tests.conftest import L, random_spectrum, relerr

ROTATION = (0.4, 0.9, 0.2)


def _rotate_flat(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(ref.rotate_spectrum(x.numpy(), *ROTATION),
                        dtype=torch.complex64)


def _rotate_so3(blocks: list[torch.Tensor]) -> list[torch.Tensor]:
    """The left action touches only the row index of each Wigner block."""
    out = []
    for l, b in enumerate(blocks):
        D = np.conj(ref.wigner_D(l, *ROTATION))
        out.append(torch.einsum("ms,...sn->...mn", torch.tensor(D, dtype=torch.complex64), b))
    return out


# ------------------------------------------------------------ equivariance

def test_so3_conv_is_left_equivariant():
    conv = SO3Conv(2, 3, L)
    x = random_spectrum(L, n=1, channels=2)
    a = conv(_rotate_flat(x))
    b = _rotate_so3(conv(x))
    assert max(relerr(p.detach().numpy(), q.detach().numpy())
               for p, q in zip(a, b)) < 1e-4


def test_spin_conv_is_left_equivariant():
    conv = SpinSO3Conv(2, 3, L)
    x = random_spectrum(L, n=1, channels=2)
    a = conv(_rotate_flat(x))
    b = _rotate_flat(conv(x).detach())
    assert relerr(a.detach().numpy(), b.numpy()) < 1e-4


# ------------------------------------------------------------------ purity

def _module_cases():
    x2 = torch.randn(1, 4, 6, 5, dtype=torch.complex64)
    x3 = torch.randn(1, 4, 6, 5, 6, dtype=torch.complex64)
    flat = random_spectrum(L, n=0, channels=4)
    blocks = [torch.randn(1, 4, 2 * l + 1, 2 * l + 1, dtype=torch.complex64)
              for l in range(L + 1)]
    return [
        ("WeightedLeakyReLU", WeightedLeakyReLU(), x2),
        ("MagnitudeNonlinearity", MagnitudeNonlinearity(n=[0, 1]), x2),
        ("PhaseCollapse", PhaseCollapseNonlinearity(n=[0, 1], nfeatures=4), x2),
        ("ComplexDropout", ComplexDropout(p=0.2).eval(), x2),
        ("SphericalBatchNorm", SphericalBatchNormalization(
            n=[0, 1], nfeatures=4, n_beta=5, dimensions=2), x2),
        ("SpinSpectralLayerNorm", SpinSpectralLayerNormalization(
            n=0, nfeatures=4, maxL=L), flat),
        ("RealNonlinearity", RealNonlinearity(), x2),
        ("SpectralDropout", SpectralDropout(p=0.2), blocks),
    ]


@pytest.mark.parametrize("name,module,x", _module_cases())
def test_module_does_not_mutate_its_input(name, module, x):
    before = [t.clone() for t in x] if isinstance(x, list) else x.clone()
    module(x)
    if isinstance(x, list):
        assert all(torch.equal(a, b) for a, b in zip(before, x))
    else:
        assert torch.equal(before, x)


@pytest.mark.parametrize("name,module,x", _module_cases()[:6])
def test_module_is_differentiable(name, module, x):
    x = [t.clone().requires_grad_(True) for t in x] if isinstance(x, list) \
        else x.clone().requires_grad_(True)
    out = module(x)
    out = torch.cat([t.flatten() for t in out]) if isinstance(out, list) else out
    out.abs().sum().backward()
    grads = x if isinstance(x, list) else [x]
    assert all(g.grad is not None and torch.isfinite(g.grad).all() for g in grads)


# ----------------------------------------------------- normalisation maths

@pytest.mark.parametrize("cls", [SO3Conv, SpinSO3Conv])
def test_kernels_are_not_initialised_to_a_constant(cls):
    """A constant kernel makes every output channel identical at step zero."""
    k = cls(4, 3, L).kernel.detach()
    assert not torch.allclose(k[:, 0], k[:, 1])
    assert k.imag.abs().max() > 0


@pytest.mark.parametrize("cls,fan_in,fan_out", [(SO3Conv, 8, 16), (SpinSO3Conv, 8, 16)])
def test_kernel_variance_follows_glorot(cls, fan_in, fan_out):
    """Trabelsi et al. 2018: E[|w|^2] = 2/(fan_in + fan_out), phase uniform."""
    torch.manual_seed(0)
    k = cls(fan_in, fan_out, 24).kernel.detach()
    assert np.isclose((k.abs() ** 2).mean().item(), 2 / (fan_in + fan_out), rtol=0.05)
    assert np.isclose(k.angle().std().item(), np.pi / np.sqrt(3), rtol=0.05)


def test_so3_to_so3_kernel_fan_grows_with_order():
    """Both the channel and the contracted row index count toward the fan."""
    torch.manual_seed(0)
    c = SO3toSO3Conv(8, 16, 8)
    for l in (2, 5, 8):
        got = (c.kernel[l].detach().abs() ** 2).mean().item()
        assert np.isclose(got, 2 / ((8 + 16) * (2 * l + 1)), rtol=0.1)


@pytest.mark.parametrize("cls", [SpinSpectralLayerNormalization,
                                 SpinSpectralBatchNormalization])
def test_spectral_variance_weights(cls):
    """Paper appendix E: order l contributes with weight 1/(2l+1).

    `order_scaling` indexes the flat spectrum with the l=0 entry dropped, so
    order l starts at l^2-1 and the whole block carries the same weight.
    """
    w = cls(n=0, nfeatures=1, maxL=6).order_scaling
    for l in range(1, 7):
        assert np.isclose(w[l * l - 1].item(), 1 / (2 * l + 1))
        assert np.isclose(w[(l + 1) * (l + 1) - 2].item(), 1 / (2 * l + 1))


def test_spherical_batchnorm_applies_eps():
    """`epsilon` is documented and was ignored; a zero signal must not divide by zero."""
    bn = SphericalBatchNormalization(n=0, nfeatures=2, n_beta=5, dimensions=2)
    out = bn(torch.zeros(3, 2, 6, 5, dtype=torch.complex64))
    assert torch.isfinite(torch.view_as_real(out)).all()


@pytest.mark.parametrize("scale,bias", [(False, True), (True, False), (False, False)])
def test_spherical_batchnorm_optional_parameters(scale, bias):
    """`use_scale=False, use_bias=True` left `y` unbound."""
    bn = SphericalBatchNormalization(n=0, nfeatures=2, n_beta=5, dimensions=2,
                                     use_scale=scale, use_bias=bias)
    bn(torch.randn(3, 2, 6, 5, dtype=torch.complex64))


def test_spherical_batchnorm_variance_is_real():
    """E[x^2] vanishes for spin != 0 (paper appendix E); E[|x|^2] does not."""
    bn = SphericalBatchNormalization(n=1, nfeatures=2, n_beta=5, dimensions=2)
    bn(torch.randn(3, 2, 6, 5, dtype=torch.complex64))
    assert not bn.running_var.is_complex()


@pytest.mark.parametrize("n", [0, 1, [0, 1]])
def test_spectral_smoothing_selects_one_column_per_order(n):
    """The l=0 block is 1x1 and has no n=1 column; that case yields zeros.

    Every branch must return [batch, features, 2l+1] so the orders can be
    concatenated downstream.
    """
    blocks = [torch.randn(1, 4, 2 * l + 1, 2 * l + 1, dtype=torch.complex64)
              for l in range(L + 1)]
    out = SpectralSmoothing(L=L, n=n)(blocks)
    assert [tuple(o.shape) for o in out] == [(1, 4, 2 * l + 1) for l in range(L + 1)]
    if n == 1:
        assert out[0].abs().max() == 0


def _blocks(l_max, channels, batch=3, zero_spin1_l0=True):
    """A list of Wigner blocks; the spin-1 half of the l=0 block is zero."""
    out = [torch.randn(batch, channels, 2 * l + 1, 2 * l + 1, dtype=torch.complex64)
           for l in range(l_max + 1)]
    if zero_spin1_l0:
        out[0][:, channels // 2:] = 0
    return out


def test_full_batchnorm_mean_is_per_channel():
    """`torch.mean` with no `dim` collapsed the mean to a scalar, so every
    channel was centred on the same value."""
    bn = FullSpectralBatchNormalization(spins=0, nfeatures=4).train()
    x = _blocks(3, 4, zero_spin1_l0=False)
    for c in range(4):
        x[0][:, c] = c + 1                  # a known, distinct mean per channel
    bn(x)
    assert bn.running_mean.shape == (4,)
    want = bn.momentum * torch.arange(1, 5, dtype=torch.float32).to(torch.complex64)
    assert torch.allclose(bn.running_mean, want, atol=1e-6)


def test_full_layernorm_leaves_spin1_alone():
    """The l=0 coefficient of a spin-1 channel is structurally zero; subtracting
    a channel mean from it would both bias the mean and break equivariance."""
    x = _blocks(3, 4)
    out = FullSpectralLayerNormalization(n=[0, 1], nfeatures=4)(x)
    assert out[0][:, 2:].abs().max() == 0
    assert out[0][:, :2].abs().max() > 0
