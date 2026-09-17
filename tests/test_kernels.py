"""The Triton kernels against the tensor path.

These need a GPU and are skipped without one, so the rest of the suite still
runs anywhere. The tensor path is the reference: it is what the kernels are
checked against, which is how a stride bug that only showed up in the pooling
topologies was caught -- the table spans every order the model has, the
coefficients of one layer only its own, and the two stride differently.
"""

from __future__ import annotations

import pytest
import torch

from modules import kernels
from modules.equivariant import SO3CNN, SO3MNISTClassifier, SO3UNet
from modules.precomp import Packed
from tests import models
from tests.conftest import L, N_BETA, relerr

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and kernels.HAVE_TRITON),
                                reason="needs a GPU with triton")

TOPOLOGIES = {
    "flat": (SO3CNN, dict(resolution=L, widths=[1, 2, 1], spins=[[0], [0, 1], [0]])),
    "unet": (SO3UNet, dict(resolutions=[L, models.INNER, L], widths=[1, 2, 2, 1],
                           spins=[[0], [0, 1], [0, 1], [0]])),
    "classifier": (SO3MNISTClassifier, dict(resolutions=[L, models.INNER, 0],
                                            widths=[1, 2, 10], spins=[[0], [0, 1], [0]])),
}


def build(cls, arch, fuse, **kwargs):
    torch.manual_seed(models.SEED)
    return cls(arch=arch, fuse=fuse, **kwargs, **models._common()).cuda().eval()


def test_the_synthesis_kernel_matches_the_tensor_sum():
    n_beta, l_max, batch, chan = 9, 6, 2, 3
    n_grid = 2 * (l_max + 1)
    blocks = [torch.randn(batch, chan, 2 * l + 1, 2 * l + 1,
                          dtype=torch.complex64, device="cuda") for l in range(l_max + 1)]
    tables = [torch.randn(n_beta, 2 * l + 1, 2 * l + 1, device="cuda") for l in range(l_max + 1)]

    want = torch.zeros((batch, chan, n_grid, n_beta, n_grid),
                       dtype=torch.complex64, device="cuda")
    mid = n_grid // 2
    for l, block in enumerate(blocks):
        want[..., mid - l:mid + l + 1, :, mid - l:mid + l + 1] += \
            block.unsqueeze(-2) * tables[l].permute(1, 0, 2)

    packed = torch.cat([b.reshape(batch, chan, -1) for b in blocks], dim=-1)
    got = kernels.synthesise(packed, Packed.build(tables).packed, l_max, n_beta, n_grid)
    # The kernel writes at the shifted position, which is where the transforms
    # want the grid; doing it afterwards would be another copy of it.
    want = torch.fft.fftshift(want, dim=(-3, -1))
    assert relerr(got.cpu().numpy(), want.cpu().numpy()) < 1e-5


def test_the_kernel_reads_the_table_at_its_own_stride():
    """A layer band-limited below the model's uses a prefix of the orders, but
    the table's rows are still as long as the model's largest order."""
    n_beta, model_l, layer_l = 7, 8, 4
    n_grid = 2 * (layer_l + 1)
    tables = [torch.randn(n_beta, 2 * l + 1, 2 * l + 1, device="cuda") for l in range(model_l + 1)]
    blocks = [torch.randn(1, 2, 2 * l + 1, 2 * l + 1,
                          dtype=torch.complex64, device="cuda") for l in range(layer_l + 1)]

    want = torch.zeros((1, 2, n_grid, n_beta, n_grid), dtype=torch.complex64, device="cuda")
    mid = n_grid // 2
    for l, block in enumerate(blocks):
        want[..., mid - l:mid + l + 1, :, mid - l:mid + l + 1] += \
            block.unsqueeze(-2) * tables[l].permute(1, 0, 2)

    packed = torch.cat([b.reshape(1, 2, -1) for b in blocks], dim=-1)
    got = kernels.synthesise(packed, Packed.build(tables).packed, layer_l, n_beta, n_grid)
    assert relerr(got.cpu().numpy(), torch.fft.fftshift(want, dim=(-3, -1)).cpu().numpy()) < 1e-5


@pytest.mark.parametrize("name", list(TOPOLOGIES))
@pytest.mark.parametrize("arch", ["s3cnn", "scnn"])
def test_a_model_gives_the_same_answer_with_the_kernels(name, arch):
    cls, kwargs = TOPOLOGIES[name]
    x = models.sample_input().cuda()
    fast, plain = build(cls, arch, True, **kwargs), build(cls, arch, False, **kwargs)
    assert relerr(fast(x).detach().cpu().numpy(), plain(x).detach().cpu().numpy()) < 1e-5


@pytest.mark.parametrize("arch", ["s3cnn", "scnn"])
def test_the_gradients_match_too(arch):
    """The backward of the synthesis is the analysis contraction, in its own
    kernel; it is wrong in ways the forward would not show."""
    cls, kwargs = TOPOLOGIES["flat"]
    x = models.sample_input().cuda()
    fast, plain = build(cls, arch, True, **kwargs), build(cls, arch, False, **kwargs)
    fast(x).abs().sum().backward()
    plain(x).abs().sum().backward()
    for (name, a), (_, b) in zip(fast.named_parameters(), plain.named_parameters()):
        assert a.grad is not None, name
        assert relerr(a.grad.cpu().numpy(), b.grad.cpu().numpy()) < 1e-5, name
