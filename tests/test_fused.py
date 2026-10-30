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
def test_a_normalisation_keeps_the_orders_packed(arch):
    """Layer norm sums over the orders and divides by the result. Per order that
    was hundreds of kernel launches; packed it is one reduction and one divide,
    and the slab path keeps the buffer the convolution wrote."""
    x = models.sample_input()
    fast, plain = pair(flat, arch=arch, use_layer_norm=True)
    assert relerr(fast(x).detach().numpy(), plain(x).detach().numpy()) < 1e-5

    for model in (fast, plain):
        model(x).abs().sum().backward()
    for (name, a), (_, b) in zip(fast.named_parameters(), plain.named_parameters()):
        assert relerr(a.grad.numpy(), b.grad.numpy()) < 1e-5, name


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


@pytest.mark.parametrize("chunk", [1, 3, N_BETA, N_BETA + 10])
def test_the_slab_size_does_not_change_the_answer_on_the_group(chunk):
    """The same for `regroup`, which integrates each slab back onto SO(3): the
    quadrature weights have to follow the slab's latitudes. They did not, and
    every test grid fit in one slab, so the first full-size scnn found it."""
    torch.manual_seed(models.SEED)
    block = flat("scnn", fuse=False).eval().layers[0]
    blocks = [torch.randn(2, 2, 2 * l + 1, 2 * l + 1, dtype=torch.complex64) for l in range(L + 1)]
    tables = models._common()["f_wigner"]
    synth = block.synthesis

    def run(chunk):
        return fused.regroup(fused.Orders(blocks), tables, block.analysis.quadrature,
                             block.sigma, block.dropout, block.analysis.L,
                             synth.n_alpha, synth.n_beta, synth.n_gamma, chunk=chunk)

    for got, whole in zip(run(chunk), run(N_BETA)):
        assert relerr(got.detach().numpy(), whole.detach().numpy()) < 1e-6


def test_the_slab_heuristic_stays_within_the_grid():
    assert 1 <= fused.slab(179, 360, 360, 30) <= 179
    assert fused.slab(20, 8, 8, 1) == 20          # a small layer takes every latitude
    assert fused.slab(179, 360, 360, 4096) == 1   # a huge one takes them one at a time


def test_a_bigger_batch_takes_thinner_slabs():
    """The grid grows with the batch and the budget does not, so the slab
    shrinks. Batching still pays -- 1.6x less per sample at batch 8 than at 1 --
    and the 256 MiB budget is a measured minimum in both directions; see the
    sweep recorded above `SLAB_BYTES`."""
    for batch in (1, 8, 32):
        assert 1 <= fused.slab(179, 360, 360, 8, batch=batch) <= 179
    assert fused.slab(179, 360, 360, 8, batch=32) < fused.slab(179, 360, 360, 8, batch=1)


@pytest.mark.parametrize("arch", LIFTING)
@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
def test_the_recompute_flag_does_not_change_gradients(arch, train):
    """`recompute=False` keeps each slab instead of rebuilding it in the
    backward pass. That is a memory-for-time trade and nothing else, so the
    gradients must be identical -- including in train mode, where dropout is
    live and draws per slab. `use_reentrant=False` restores the RNG state
    around the recompute, so both paths see the same masks."""
    grads = []
    for recompute in (True, False):
        torch.manual_seed(models.SEED)
        model = flat(arch, fuse=True, recompute=recompute).train(train)
        torch.manual_seed(1234)                  # same dropout masks either way
        model(models.sample_input()).abs().sum().backward()
        grads.append({n: p.grad for n, p in model.named_parameters()})

    kept, rebuilt = grads
    assert kept, "no parameters compared"
    for name, a in kept.items():
        assert a is not None, name
        assert relerr(a.numpy(), rebuilt[name].numpy()) < 1e-6, name


def test_the_recompute_flag_reaches_every_block():
    """It is set on the network and read by each block, like `fuse`."""
    for recompute in (True, False):
        model = unet("s3cnn", fuse=True, recompute=recompute)
        assert model.recompute is recompute
        blocks = list(model.down) + list(model.up)
        assert blocks, "no blocks to check"
        assert all(b.recompute is recompute for b in blocks)


def test_recompute_false_actually_skips_the_checkpoint(monkeypatch):
    """The gradient tests above would pass even if the flag were ignored, since
    both paths agree by construction. This one pins that it is honoured: the
    checkpoint is what costs the second synthesis, so it must not be entered."""
    calls = []
    real = fused.checkpoint

    def spy(fn, *args, **kwargs):
        calls.append(fn)
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(fused, "checkpoint", spy)
    # `_run` only checkpoints when a tensor it was handed requires grad, and a
    # block that reads straight from the data is handed one that does not. The
    # input carries grad here so the path is actually reached.
    x = models.sample_input().requires_grad_(True)
    for recompute, want in ((True, True), (False, False)):
        calls.clear()
        torch.manual_seed(models.SEED)
        model = flat("s3cnn", fuse=True, recompute=recompute).train(True)
        model(x).abs().sum().backward()
        assert bool(calls) is want, \
            f"recompute={recompute}: checkpoint called {len(calls)} times"
