"""The SO(3) layer as one pass over slabs of latitudes.

"Fused" is the term from GPU programming for combining several operations into
one pass so that the values flowing between them stay in registers or cache and
never travel out to memory and back. Here the operations being fused are the
synthesis, the nonlinearity, the dropout and the integral over gamma, which
unfused hand each other whole copies of the SO(3) grid.

Unfused, a lifting layer walks through three tensors the size of the SO(3)
grid -- `[batch, channels, n_alpha, n_beta, n_gamma]`, gigabytes at production
resolution -- and keeps the last one alive for the backward pass, all so that a
pointwise nonlinearity can be applied and the gamma axis immediately integrated
away.

Nothing after the beta-synthesis couples different latitudes: the transforms
over alpha and gamma, the nonlinearity, the dropout and the integral over gamma
all act within one beta. So the whole chain runs on a slab of latitudes at a
time and the grid never exists at full size. Each slab is recomputed in the
backward pass rather than stored (`torch.utils.checkpoint`).

Beta is the axis to cut, not gamma: evaluating a single gamma needs every
coefficient in `n`, so a gamma slab would still require the whole grid.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from . import kernels

#: Target bytes for one slab's grid; the slab count follows from the layer size.
SLAB_BYTES = 256 * 2**20

BYTES_COMPLEX64 = 8


def slab(n_beta: int, n_alpha: int, n_gamma: int, channels: int, batch: int = 1) -> int:
    """How many latitudes to take at once, at least one and at most all."""
    per_beta = batch * channels * n_alpha * n_gamma * BYTES_COMPLEX64
    return max(1, min(n_beta, SLAB_BYTES // max(per_beta, 1)))


class Orders:
    """The lifted coefficients, order by order.

    A `list` of blocks is one already in hand. A convolution and its input is
    the same thing unevaluated: each order is computed when the synthesis asks
    for it and dropped straight after, so the lifted signal -- which is several
    times the size of what goes into the convolution -- never accumulates.
    """

    def __init__(self, source, x=None):
        self.source = source
        self.materialised = x is None
        #: A group-to-group convolution takes one block per order, not one tensor.
        self.listed = isinstance(x, list)
        self.x = () if x is None else (tuple(x) if self.listed else (x,))
        self.L = (len(source) - 1) if self.materialised else source.L

    @property
    def leading(self) -> torch.Tensor:
        """The l = 0 block, for its batch, channels, dtype and device. It is 1x1."""
        return self[0]

    def __getitem__(self, l: int) -> torch.Tensor:
        if self.materialised:
            return self.source[l]
        return self.source.block(l, list(self.x) if self.listed else self.x[0])

    def tensors(self) -> tuple:
        """What the recompute has to be told about, for autograd."""
        return tuple(self.source) if self.materialised else self.x

    def rebind(self, tensors: tuple) -> "Orders":
        """The same orders over the tensors handed back by the recompute."""
        if self.materialised:
            return Orders(list(tensors))
        return Orders(self.source, list(tensors) if self.listed else tensors[0])

    def packed(self) -> torch.Tensor:
        """Every order in one buffer, `[batch, channels, sum (2l+1)^2]`.

        The kernel walks the orders in place from this, so the lifted signal is
        built once per layer rather than once per slab.
        """
        blocks = [self[l] for l in range(self.L + 1)]
        return torch.cat([b.reshape(*b.shape[:2], -1) for b in blocks], dim=-1)


def _accumulate(orders: "Orders", tables, betas: slice, n_alpha: int, n_gamma: int):
    """The beta-synthesis, order by order, each adding into a sub-block.

    The kernel in `modules/kernels.py` computes the same sum the other way
    round, one output cell at a time, which is several times faster. This stays
    as the reference and as the path off the GPU.
    """
    head = orders.leading
    out = torch.zeros((*head.shape[:2], n_alpha, betas.stop - betas.start, n_gamma),
                      dtype=torch.complex64, device=head.device)
    mid_a, mid_g = n_alpha // 2, n_gamma // 2
    for l in range(orders.L + 1):
        ww = tables[l][betas].permute(1, 0, 2)          # [m, beta, n]
        out[..., mid_a - l:mid_a + l + 1, :, mid_g - l:mid_g + l + 1] += \
            orders[l].unsqueeze(-2) * ww
    return out


def _synthesise(source, tables, betas: slice, n_alpha: int, n_gamma: int, l_max=None):
    """Coefficients -> grid values, for the latitudes in `betas`.

    `[batch, channels, n_alpha, |betas|, n_gamma]`, the same arithmetic as
    `SO3iFFT` restricted to a slab. A packed tensor goes through the kernel, an
    `Orders` through the tensor path.
    """
    if isinstance(source, Orders):
        out = _accumulate(source, tables, betas, n_alpha, n_gamma)
    else:
        out = kernels.synthesise(source, tables.packed[betas], l_max,
                                 betas.stop - betas.start, n_alpha)
    out = torch.fft.fftshift(out, dim=(-3, -1))
    out = torch.fft.fft(out, dim=-3, norm="ortho")
    return torch.fft.fft(out, dim=-1)


def _source(orders: "Orders", tables):
    """What the slabs read: a packed buffer when the kernel can run, else the orders."""
    if kernels.available(orders.leading) and hasattr(tables, "packed"):
        packed = orders.packed()
        return packed, (packed,), orders.L
    return orders, orders.tensors(), orders.L


def _analyse(grid, tables, rule, betas: slice, L: int, n_beta: int, into: list):
    """Grid values -> coefficients, accumulating one slab's contribution.

    The beta integral is a sum, so each slab adds its own part; the same
    arithmetic as `SO3FFT` restricted to a slab.
    """
    n_alpha, n_gamma = grid.size(-3), grid.size(-1)
    maf_a, maf_g = (n_alpha - 1) // 2, (n_gamma - 1) // 2
    grid = torch.fft.fft(grid, dim=-3, norm="ortho")
    grid = torch.fft.fft(grid, dim=-1, norm="forward")
    grid = torch.fft.fftshift(grid, dim=(-3, -1))
    grid = torch.flip(grid, dims=(-3, -1))
    for l in range(L + 1):
        A = rule.operator(tables[l][betas], l, n_beta).permute(1, 0, 2)
        chunk = grid[..., maf_a - l:maf_a + l + 1, :, maf_g - l:maf_g + l + 1]
        into[l] = into[l] + (chunk * A).sum(dim=-2)
    return into


def _run(fn, *tensors):
    """Recompute in the backward pass instead of keeping the slab alive."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return checkpoint(fn, *tensors, use_reentrant=False)
    return fn(*tensors)


def project(orders: "Orders", tables, sigma, dropout, smoothing,
            n_alpha: int, n_beta: int, n_gamma: int, chunk: int | None = None):
    """Synthesis -> sigma -> dropout -> integral over gamma, slab by slab.

    Returns the S2 grid `[batch, channels, n_alpha, n_beta]`. This is the path
    of every architecture that projects back to the sphere after the
    nonlinearity.
    """
    head = orders.leading
    source, tensors, l_max = _source(orders, tables)
    step = chunk or slab(n_beta, n_alpha, n_gamma, head.size(1), head.size(0))
    out = torch.empty((*head.shape[:2], n_alpha, n_beta),
                      dtype=torch.complex64, device=head.device)
    for start in range(0, n_beta, step):
        betas = slice(start, min(start + step, n_beta))

        def one(*passed, betas=betas):
            here = orders.rebind(passed) if isinstance(source, Orders) else passed[0]
            return smoothing(dropout(sigma(
                _synthesise(here, tables, betas, n_alpha, n_gamma, l_max))))

        out[..., betas] = _run(one, *tensors)
    return out


def regroup(orders: "Orders", tables, rule, sigma, dropout, L: int,
            n_alpha: int, n_beta: int, n_gamma: int, chunk: int | None = None):
    """Synthesis -> sigma -> dropout -> analysis, slab by slab, staying on SO(3).

    Returns coefficients, one block per order. This is the path of an
    architecture that never projects back to the sphere.
    """
    head = orders.leading
    source, tensors, l_max = _source(orders, tables)
    step = chunk or slab(n_beta, n_alpha, n_gamma, head.size(1), head.size(0))
    out = [torch.zeros((*head.shape[:2], 2 * l + 1, 2 * l + 1),
                       dtype=torch.complex64, device=head.device) for l in range(L + 1)]
    for start in range(0, n_beta, step):
        betas = slice(start, min(start + step, n_beta))

        def one(*passed, betas=betas):
            here = orders.rebind(passed) if isinstance(source, Orders) else passed[0]
            grid = _synthesise(here, tables, betas, n_alpha, n_gamma, l_max)
            grid = dropout(sigma(grid))
            empty = [torch.zeros((), dtype=torch.complex64, device=grid.device)] * (L + 1)
            return tuple(_analyse(grid, tables, rule, betas, L, n_beta, empty))

        for l, part in enumerate(_run(one, *tensors)):
            out[l] = out[l] + part
    return out
