"""The slab-by-slab SO(3) layer against the plain one.

`fuse=False` keeps the original path -- synthesise the whole grid, apply sigma,
integrate -- so the two can be run side by side. They must agree on outputs and
on gradients; the slab version only changes when the work happens, not what it
computes. Dropout is the one exception, and it draws per slab, so these run in
eval mode where it is the identity.
"""

from __future__ import annotations

import pytest
import torch

from modules import fused
from modules.equivariant import SO3CNN, SO3MNISTClassifier, SO3UNet
from tests import models
from tests.conftest import L, N_BETA, relerr

LIFTING = ["s3cnn", "scnn"]


def pair(build, **kwargs):
    """The same model twice, fused and not, from one seed."""
    out = []
    for fuse in (True, False):
        torch.manual_seed(models.SEED)
        out.append(build(fuse=fuse, **kwargs).eval())
    return out


def flat(arch, **kw):
    return SO3CNN(arch=arch, resolution=L, widths=[1, 2, 1],
                  spins=[[0], [0, 1], [0]], **kw, **models._common())


def unet(arch, **kw):
    return SO3UNet(arch=arch, resolutions=[L, models.INNER, L], widths=[1, 2, 2, 1],
                   spins=[[0], [0, 1], [0, 1], [0]], **kw, **models._common())


def classifier(arch, **kw):
    return SO3MNISTClassifier(arch=arch, resolutions=[L, models.INNER, 0],
                              widths=[1, 2, 10], spins=[[0], [0, 1], [0]],
                              **kw, **models._common())


@pytest.mark.parametrize("build", [flat, unet, classifier],
                         ids=["flat", "unet", "classifier"])
@pytest.mark.parametrize("arch", LIFTING)
def test_slabs_compute_what_the_whole_grid_computes(build, arch):
    x = models.sample_input()
    fast, plain = pair(build, arch=arch)
    assert relerr(fast(x).detach().numpy(), plain(x).detach().numpy()) < 1e-5


@pytest.mark.parametrize("arch", LIFTING)
def test_gradients_survive_the_recompute(arch):
    """Each slab is recomputed in the backward pass rather than stored, which is
    where the memory goes; the gradients have to come out the same anyway."""
    x = models.sample_input()
    fast, plain = pair(flat, arch=arch)
    for model in (fast, plain):
        model(x).abs().sum().backward()
    for (name, a), (_, b) in zip(fast.named_parameters(), plain.named_parameters()):
        assert a.grad is not None, name
        assert relerr(a.grad.numpy(), b.grad.numpy()) < 1e-5, name


@pytest.mark.parametrize("chunk", [1, 3, N_BETA, N_BETA + 10])
def test_the_slab_size_does_not_change_the_answer(chunk):
    """One latitude at a time, all of them at once, or more than there are."""
    x = models.sample_input()
    torch.manual_seed(models.SEED)
    model = flat("s3cnn", fuse=False).eval()
    want = model(x)

    blocks = [torch.randn(2, 2, 2 * l + 1, 2 * l + 1, dtype=torch.complex64) for l in range(L + 1)]
    tables = models._common()["f_wigner"]
    sigma, dropout, smoothing = (model.layers[0].sigma, model.layers[0].dropout,
                                 model.layers[0].smoothing)
    synth = model.layers[0].synthesis
    got = fused.project(fused.Orders(blocks), tables, sigma, dropout, smoothing,
                        synth.n_alpha, synth.n_beta, synth.n_gamma, chunk=chunk)
    whole = fused.project(fused.Orders(blocks), tables, sigma, dropout, smoothing,
                          synth.n_alpha, synth.n_beta, synth.n_gamma, chunk=N_BETA)
    assert relerr(got.detach().numpy(), whole.detach().numpy()) < 1e-6


def test_the_slab_heuristic_stays_within_the_grid():
    assert fused.slab(179, 360, 360, 30) >= 1
    assert fused.slab(179, 360, 360, 30) <= 179
    assert fused.slab(20, 8, 8, 1) == 20          # a small layer takes every latitude
    assert fused.slab(179, 360, 360, 4096) == 1   # a huge one takes them one at a time
