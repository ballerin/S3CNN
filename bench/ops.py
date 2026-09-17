"""Per-operator timings: the transforms, the convolutions, the smoothing and
the nonlinearity.

Wigner tables are synthetic tensors of the right shape and dtype. None of
these kernels branches on its data, so the timings are the same as with the
real tables, and the suite then runs without the `precomp/` directory at all.
Correctness is the test suite's job, not this one's.
"""

from __future__ import annotations

import torch

from bench.harness import BYTES_COMPLEX64 as CX
from bench.harness import Result, time_call
from modules.SO3ConvLayer import SO3Conv, SpinSO3Conv
from modules.SO3fft import S2FFT, S2iFFT, SO3FFT, SO3iFFT
from modules.SpectralModules import MagnitudeNonlinearity, SpatialSmoothing

SIZES = {
    # name: (L, n_beta, channel sweep) -- "prod" is what every shipped ERA5
    # config trains at; "small" is for checking the harness itself.
    "small": (59, 59, [2, 8]),
    "prod": (179, 179, [2, 8, 30]),
}


def _angles(L):
    return 2 * (L + 1)


def _so3_spectrum(L, c, batch, device):
    return [torch.randn(batch, c, 2 * l + 1, 2 * l + 1,
                        dtype=torch.complex64, device=device) for l in range(L + 1)]


def full_table(L, n_beta, device):
    """The d^l blocks the SO(3) transforms index; 11.1 GB at L = 179."""
    return [torch.randn(n_beta, 2 * l + 1, 2 * l + 1,
                        dtype=torch.complex64, device=device) for l in range(L + 1)]


def slice_tables(L, n_beta, device):
    """Columns n = 0 and n = 1 of each block, which is all the S2 path reads."""
    n0 = [torch.randn(n_beta, 2 * l + 1, dtype=torch.complex64, device=device)
          for l in range(L + 1)]
    n1 = [torch.randn(n_beta, 2 * l + 1, dtype=torch.complex64, device=device)
          for l in range(1, L + 1)]
    return n0, n1


def _sizes(L, n_beta, c, batch):
    a = _angles(L)
    blocks = sum((2 * l + 1) ** 2 for l in range(L + 1))
    return dict(
        table_full=blocks * n_beta * CX,
        table_slice=(L + 1) ** 2 * n_beta * CX,
        spec_so3=batch * c * blocks * CX,
        spec_s2=batch * c * (L + 1) ** 2 * CX,
        grid_so3=batch * c * a * n_beta * a * CX,
        grid_s2=batch * c * a * n_beta * CX,
    )


def cases(size: str, batch: int, iters: int, only: str | None = None):
    """Yield one Result per operator and channel count."""
    L, n_beta, channels = SIZES[size]
    a = _angles(L)
    dev = torch.device("cuda")
    full = torch.nested.nested_tensor(full_table(L, n_beta, dev))
    n0, n1 = slice_tables(L, n_beta, dev)
    slices = dict(d_matrices_n0=n0, d_matrices_n1=n1)

    for c in channels:
        s = _sizes(L, n_beta, c, batch)
        p = dict(L=L, C=c, B=batch)
        spec_so3 = _so3_spectrum(L, c, batch, dev)
        spec_s2 = torch.randn(batch, c, (L + 1) ** 2, dtype=torch.complex64, device=dev)
        grid_s2 = torch.randn(batch, c, a, n_beta, dtype=torch.complex64, device=dev)

        so3_ifft = SO3iFFT(L=L, n_alpha=a, n_beta=n_beta, n_gamma=a)
        so3_fft = SO3FFT(L=L)
        s2_ifft = S2iFFT(L=L, n=0, n_alpha=a, n_beta=n_beta)
        s2_fft = S2FFT(L=L, n=0)
        so3_conv = SO3Conv(c, c, L).to(dev)
        spin_conv = SpinSO3Conv(c, c, L).to(dev)
        smoothing = SpatialSmoothing(n=0)
        nonlinearity = MagnitudeNonlinearity(n=0).to(dev)
        grid_so3 = so3_ifft(spec_so3, d_matrices=full)

        ops = [
            ("SO3iFFT", lambda: so3_ifft(spec_so3, d_matrices=full),
             s["table_full"] + s["spec_so3"] + s["grid_so3"]),
            ("SO3FFT", lambda: so3_fft(grid_so3, d_matrices=full),
             s["table_full"] + s["grid_so3"] + s["spec_so3"]),
            ("S2iFFT", lambda: s2_ifft(spec_s2, **slices),
             s["table_slice"] + s["spec_s2"] + s["grid_s2"]),
            ("S2FFT", lambda: s2_fft(grid_s2, **slices),
             s["table_slice"] + s["grid_s2"] + s["spec_s2"]),
            ("SO3Conv", lambda: so3_conv(spec_s2),
             s["spec_s2"] + s["spec_so3"]),
            ("SpinSO3Conv", lambda: spin_conv(spec_s2),
             2 * s["spec_s2"]),
            ("SpatialSmoothing", lambda: smoothing(grid_so3),
             s["grid_so3"] + s["grid_s2"]),
            ("Nonlinearity", lambda: nonlinearity(grid_s2),
             2 * s["grid_s2"]),
        ]

        for name, fn, traffic in ops:
            if only is not None and only not in name:
                continue
            ms, std, peak = time_call(fn, iters=iters)
            yield Result("ops", name, p, ms, std, peak, traffic)
            torch.cuda.empty_cache()

        del spec_so3, spec_s2, grid_s2, grid_so3
        torch.cuda.empty_cache()
