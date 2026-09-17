"""Triton kernels for the beta-synthesis and its transpose.

The synthesis sums, for every latitude, the coefficients of all orders that
reach a given `(m, n)`:

    grid[m, beta, n] = sum_{l >= max(|m|,|n|)} x[l, m, n] * d[l, beta, m, n]

Written with tensor operations this becomes one pass per order, each adding
into a sub-block of the output, so a cell near the middle is read and written
about L times over -- tens of gigabytes of traffic for a grid of one. Padding
the orders into a rectangle lets a single matmul do it, at the price of a
padded copy of the Wigner table, which is roughly three times its size at the
resolutions used here.

A kernel needs neither: the orders live in one buffer at offset
`l(4l^2-1)/3`, so a program can walk them in place and keep its running sum in
registers, writing each output cell exactly once.

The tables are real, so a complex multiply is two independent accumulations
rather than four, and the imaginary part rides along at no extra table traffic.
Complex values are read and written where they already are, interleaved as
(re, im) pairs: splitting them into separate planes meant copying the whole
gradient grid twice per slab, which cost more than the arithmetic.

The grid the transforms want is `fftshift`ed, so the kernels write and read at
the shifted position directly. Done afterwards it is a `roll`, i.e. another
full copy of the grid.

`grad` of the synthesis is a sum over latitudes at fixed order, which is the
analysis contraction; the two kernels here are therefore each other's backward.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                  # pragma: no cover - env without triton
    HAVE_TRITON = False


def available(*tensors) -> bool:
    """Whether the kernels can run on these tensors."""
    return HAVE_TRITON and all(t.is_cuda for t in tensors)


def base_offset(l: int) -> int:
    """Where order `l` starts in a packed triangular buffer: sum_{k<l} (2k+1)^2."""
    return l * (4 * l * l - 1) // 3


if HAVE_TRITON:

    def _tiles():
        """Tile shapes worth trying: the accumulator lives in registers, so the
        product of the blocks is what decides whether this spills."""
        out = []
        for c, m, b, n in [(1, 8, 4, 32), (2, 8, 2, 32), (4, 4, 2, 32), (1, 16, 2, 64),
                           (2, 4, 4, 64), (4, 8, 1, 64), (8, 4, 1, 32), (1, 8, 8, 32),
                           (1, 4, 2, 64), (2, 16, 1, 64)]:
            for warps in (4, 8):
                out.append(triton.Config(
                    {"BLOCK_C": c, "BLOCK_M": m, "BLOCK_B": b, "BLOCK_N": n},
                    num_warps=warps, num_stages=2))
        return out

    @triton.autotune(configs=_tiles(), key=["n_total", "tab_total", "n_chan", "l_max", "n_beta"])
    @triton.jit
    def _synthesis(x, table, out, n_total, tab_total, n_chan, l_max, n_beta, n_grid,
                   BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr):
        """One tile of `grid[channel, m, beta, n]`, summed over orders in registers.

        The tile spans several channels so that each order's slice of the table
        is read once for all of them: the table is the largest thing moving, and
        reading it per channel is what makes the tensor version bandwidth-bound.
        """
        c0 = tl.program_id(0) * BLOCK_C
        m0 = tl.program_id(1) * BLOCK_M
        rest = tl.program_id(2)
        n_tiles = (2 * l_max + 1 + BLOCK_N - 1) // BLOCK_N
        b0 = (rest // n_tiles) * BLOCK_B
        n0 = (rest % n_tiles) * BLOCK_N

        c = c0 + tl.arange(0, BLOCK_C)
        m = m0 + tl.arange(0, BLOCK_M)                    # padded index, 0 .. 2L
        n = n0 + tl.arange(0, BLOCK_N)
        beta = b0 + tl.arange(0, BLOCK_B)
        c_ok = c < n_chan
        m_ok = m < 2 * l_max + 1
        n_ok = n < 2 * l_max + 1
        b_ok = beta < n_beta

        # No order below this one reaches the tile, so start there.
        m_a, m_b = m0 - l_max, m0 + BLOCK_M - 1 - l_max
        n_a, n_b = n0 - l_max, n0 + BLOCK_N - 1 - l_max
        m_a, m_b = tl.where(m_a < 0, -m_a, m_a), tl.where(m_b < 0, -m_b, m_b)
        n_a, n_b = tl.where(n_a < 0, -n_a, n_a), tl.where(n_b < 0, -n_b, n_b)
        m_lo = tl.where((m0 <= l_max) & (m0 + BLOCK_M - 1 >= l_max), 0, tl.minimum(m_a, m_b))
        n_lo = tl.where((n0 <= l_max) & (n0 + BLOCK_N - 1 >= l_max), 0, tl.minimum(n_a, n_b))
        l_start = tl.maximum(m_lo, n_lo)

        acc_re = tl.zeros((BLOCK_C, BLOCK_B, BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_im = tl.zeros((BLOCK_C, BLOCK_B, BLOCK_M, BLOCK_N), dtype=tl.float32)

        for l in range(l_start, l_max + 1):
            size = 2 * l + 1
            base = l * (4 * l * l - 1) // 3
            row = m - (l_max - l)                          # 0 .. 2l within the order
            col = n - (l_max - l)
            in_m = (row >= 0) & (row < size) & m_ok
            in_n = (col >= 0) & (col < size) & n_ok
            off = base + row[:, None] * size + col[None, :]
            keep = in_m[:, None] & in_n[None, :]

            d = tl.load(table + beta[:, None, None] * tab_total + off[None, :, :],
                        mask=b_ok[:, None, None] & keep[None, :, :], other=0.0)
            at = 2 * (c[:, None, None] * n_total + off[None, :, :])
            take = c_ok[:, None, None] & keep[None, :, :]
            xr = tl.load(x + at, mask=take, other=0.0)
            xi = tl.load(x + at + 1, mask=take, other=0.0)
            acc_re += xr[:, None, :, :] * d[None, :, :, :]
            acc_im += xi[:, None, :, :] * d[None, :, :, :]

        # The orders fill every row and column but one, which the caller zeroes.
        # Writing at the fftshifted position saves rolling the grid afterwards.
        half = n_grid // 2
        m_at = (m + 1 + half) % n_grid
        n_at = (n + 1 + half) % n_grid
        idx = (c[:, None, None, None] * n_grid * n_beta * n_grid
               + m_at[None, None, :, None] * n_beta * n_grid
               + beta[None, :, None, None] * n_grid
               + n_at[None, None, None, :])
        store = (c_ok[:, None, None, None] & b_ok[None, :, None, None]
                 & m_ok[None, None, :, None] & n_ok[None, None, None, :])
        tl.store(out + 2 * idx, acc_re, mask=store)
        tl.store(out + 2 * idx + 1, acc_im, mask=store)

    @triton.jit
    def _analysis(g, table, out, n_total, tab_total, l_max, n_beta, n_grid,
                  BLOCK: tl.constexpr):
        """One tile of packed coefficients, summed over latitudes."""
        bc = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = offs < n_total

        # Which order owns each offset, by walking the closed-form starts.
        l = tl.zeros((BLOCK,), dtype=tl.int32)
        for k in range(1, l_max + 1):
            l = tl.where(offs >= k * (4 * k * k - 1) // 3, k, l)
        size = 2 * l + 1
        inner = offs - l * (4 * l * l - 1) // 3
        row = inner // size
        col = inner % size

        half = n_grid // 2                              # into the shifted grid
        m = (row + (l_max - l) + 1 + half) % n_grid
        n = (col + (l_max - l) + 1 + half) % n_grid

        acc_re = tl.zeros((BLOCK,), dtype=tl.float32)
        acc_im = tl.zeros((BLOCK,), dtype=tl.float32)
        for b in range(0, n_beta):
            d = tl.load(table + b * tab_total + offs, mask=live, other=0.0)
            idx = 2 * (bc * n_grid * n_beta * n_grid + m * n_beta * n_grid
                       + b * n_grid + n)
            gr = tl.load(g + idx, mask=live, other=0.0)
            gi = tl.load(g + idx + 1, mask=live, other=0.0)
            acc_re += gr * d
            acc_im += gi * d

        at = 2 * (bc * n_total + offs)
        tl.store(out + at, acc_re, mask=live)
        tl.store(out + at + 1, acc_im, mask=live)


class Synthesis(torch.autograd.Function):
    """Packed coefficients -> grid, with the transpose for the backward pass."""

    @staticmethod
    def forward(ctx, packed, table, l_max: int, n_beta: int, n_grid: int):
        bc = packed.shape[0] * packed.shape[1]
        n_total = packed.shape[-1]
        x = packed.reshape(bc, n_total).contiguous()
        out = torch.empty((bc, n_grid, n_beta, n_grid),
                          dtype=torch.complex64, device=packed.device)
        # Every cell is written except one row and one column, which no order
        # reaches; after the shift they sit in the middle.
        half = n_grid // 2
        out[:, half] = 0
        out[:, :, :, half] = 0
        def grid(meta):
            return (triton.cdiv(bc, meta["BLOCK_C"]),
                    triton.cdiv(2 * l_max + 1, meta["BLOCK_M"]),
                    triton.cdiv(n_beta, meta["BLOCK_B"])
                    * triton.cdiv(2 * l_max + 1, meta["BLOCK_N"]))

        _synthesis[grid](torch.view_as_real(x), table, torch.view_as_real(out),
                         n_total, table.shape[-1], bc, l_max, n_beta, n_grid)
        ctx.save_for_backward(table)
        ctx.sizes = (packed.shape, l_max, n_beta, n_grid)
        return out.reshape(*packed.shape[:2], n_grid, n_beta, n_grid)

    @staticmethod
    def backward(ctx, grad):
        table, = ctx.saved_tensors
        shape, l_max, n_beta, n_grid = ctx.sizes
        bc, n_total = shape[0] * shape[1], shape[-1]
        grad = grad.reshape(bc, n_grid, n_beta, n_grid).contiguous()
        out = torch.empty((bc, n_total), dtype=torch.complex64, device=grad.device)
        grid = (bc, triton.cdiv(n_total, 256))
        _analysis[grid](torch.view_as_real(grad), table, torch.view_as_real(out),
                        n_total, table.shape[-1], l_max, n_beta, n_grid,
                        BLOCK=256, num_warps=4)
        return out.reshape(shape), None, None, None, None


def synthesise(packed, table, l_max: int, n_beta: int, n_grid: int):
    """`[batch, channels, packed]` -> `[batch, channels, n_grid, n_beta, n_grid]`.

    The result is already `fftshift`ed on the two grid axes, ready for the
    transforms over alpha and gamma.
    """
    return Synthesis.apply(packed, table, l_max, n_beta, n_grid)
