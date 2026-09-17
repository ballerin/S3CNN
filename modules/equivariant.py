"""One equivariant layer and the four networks assembled from it.

Every network here is a stack of the same `Block`; they differ only in how the
blocks are wired -- flat, UNet, autoencoder, classifier -- and `arch` decides
what happens inside each block (see `modules/architectures.py`).

Spectra are flat `[B, C, (L+1)^2]` tensors on S2 and per-order lists on SO(3);
the index convention is documented at the top of `SO3fft.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fused, precomp
from .SO3fft import S2FFT, S2iFFT, SO3FFT, SO3iFFT, SO3_to_S2
from .SO3lib import normalise_spins
from .architectures import resolve
from .SO3ConvLayer import SO3Conv, SO3toSO3Conv, SpinSO3Conv
from .SpectralModules import ComplexDropout, MagnitudeNonlinearity, PhaseCollapseNonlinearity, \
    SpatialSmoothing, WeightedLeakyReLU, WeightedReLU, \
    FullSpectralBatchNormalization, FullSpectralLayerNormalization, \
    SpinSpectralBatchNormalization, SpinSpectralLayerNormalization

DROPOUT = 0.2


def n_coefficients(L: int) -> int:
    """Length of a flat spectrum band-limited at L."""
    return (L + 1) ** 2


def resize(x, L: int):
    """Truncate or zero-pad a spectrum to band-limit L: spectral (un)pooling.

    On S2 the spectrum is one flat tensor, so this is a pad or a slice. On SO(3)
    it is one `[B, C, 2l+1, 2l+1]` block per order, so pooling drops the blocks
    above L and unpooling appends zero ones. Truncation is the pooling the whole
    package uses; Cohen et al. 2018 instead max-pool on the group grid.
    """
    if not isinstance(x, list):
        return F.pad(x, (0, n_coefficients(L) - x.size(-1)))
    head = x[0]
    return x[:L + 1] + [torch.zeros((*head.shape[:2], 2 * l + 1, 2 * l + 1),
                                    dtype=head.dtype, device=head.device)
                        for l in range(len(x), L + 1)]


def concatenate(skip, x):
    """Join a skip connection to a signal at its band-limit, along the channels."""
    if isinstance(skip, list):
        return [torch.cat([s, t], dim=1) for s, t in zip(skip, resize(x, len(skip) - 1))]
    return torch.cat([skip, resize(x, n_orders(skip) - 1)], dim=1)


def n_orders(x: torch.Tensor) -> int:
    """Number of orders in a flat spectrum: L + 1."""
    return int(round(x.size(-1) ** 0.5))


class Block(nn.Module):
    """conv -> [norm] -> synthesis -> sigma -> dropout -> [smoothing] -> analysis.

    The convolution runs at band-limit `L`; the analysis at `L_out`, which pools
    when it is lower. With `conv_only` the block stops after the convolution,
    which is how every network ends.
    """

    def __init__(self, net: "EquivariantNetwork", f_in: int, f_out: int, spins: list,
                 L: int, L_out: int | None = None, lifted_input: bool = False,
                 conv_only: bool = False):
        super().__init__()
        spec = net.spec
        self.lifts, self.projects, self.conv_only = spec.lifts, spec.projects, conv_only
        self.fuse = net.fuse

        if spec.filters == "degree":
            self.conv = SpinSO3Conv(nfeature_in=f_in, nfeature_out=f_out, L=L)
        elif lifted_input:
            self.conv = SO3toSO3Conv(nfeature_in=f_in, nfeature_out=f_out, L=L)
        else:
            self.conv = SO3Conv(nfeature_in=f_in, nfeature_out=f_out, L=L)
        if conv_only:
            return

        n_alpha = 2 * (L + 1)
        if self.lifts:
            self.synthesis = SO3iFFT(L=L, n_alpha=n_alpha, n_beta=net.n_beta, n_gamma=n_alpha)
        else:
            self.synthesis = S2iFFT(L=L, n_alpha=n_alpha, n_beta=net.n_beta, n=spins)

        self.batchnorm = self.layernorm = None
        if net.use_batch_norm:
            self.batchnorm = (FullSpectralBatchNormalization(spins=spins, nfeatures=f_out)
                              if self.lifts else
                              SpinSpectralBatchNormalization(nfeatures=f_out, n=spins))
        if net.use_layer_norm:
            self.layernorm = (FullSpectralLayerNormalization(n=spins, nfeatures=f_out)
                              if self.lifts else
                              SpinSpectralLayerNormalization(nfeatures=f_out, n=spins))

        self.sigma = {
            "relu": lambda: WeightedReLU(),
            "leaky_relu": lambda: WeightedLeakyReLU(),
            "magnitude": lambda: MagnitudeNonlinearity(n=spins),
            "phase_collapse": lambda: PhaseCollapseNonlinearity(n=spins, nfeatures=f_out),
        }[net.nonlinearity_type]()
        self.dropout = ComplexDropout(p=DROPOUT)
        self.smoothing = SpatialSmoothing(n=spins) if self.projects else None

        # Without the projection an SO(3) signal stays on the group (scnn).
        L_out = L if L_out is None else L_out
        self.analysis = (SO3FFT(L=L_out) if self.lifts and not self.projects
                         else S2FFT(L=L_out, n=spins))

    def forward(self, x, net: "EquivariantNetwork"):
        if self.conv_only:
            return self.conv(x)

        normed = self.batchnorm is not None or self.layernorm is not None
        if self.lifts and self.fuse and not normed:
            # Nothing between the convolution and the synthesis, so the orders
            # can be convolved as the synthesis asks for them and the lifted
            # signal never accumulates.
            return self._fused(fused.Orders(self.conv, x), net)

        x = self.conv(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.layernorm is not None:
            x = self.layernorm(x)
        if self.lifts and self.fuse:
            return self._fused(fused.Orders(x), net)
        x = (self.synthesis(x, d_matrices=net.get_buffered_full_wigner()) if self.lifts
             else self.synthesis(x, **net.slices()))
        x = self.dropout(self.sigma(x))
        if self.smoothing is not None:
            x = self.smoothing(x)
        if isinstance(self.analysis, SO3FFT):
            return self.analysis(x, d_matrices=net.get_buffered_full_wigner())
        return self.analysis(x, **net.slices())

    def _fused(self, orders, net: "EquivariantNetwork"):
        """The same chain, one slab of latitudes at a time; see `modules/fused.py`."""
        tables = net.get_buffered_full_wigner()
        synth = self.synthesis
        if self.projects:
            grid = fused.project(orders, tables, self.sigma, self.dropout, self.smoothing,
                                 synth.n_alpha, synth.n_beta, synth.n_gamma)
            return self.analysis(grid, **net.slices())
        return fused.regroup(orders, tables, self.analysis.quadrature, self.sigma, self.dropout,
                             self.analysis.L, synth.n_alpha, synth.n_beta, synth.n_gamma)


class EquivariantNetwork(nn.Module):
    """What every topology shares: the architecture, the Wigner tables, and the
    transforms between the input grid and the first and last spectra.

    Tables are loaded from `precomp/` unless passed in; they are buffers, so they
    follow `.to(device)` but stay out of the state_dict.
    """

    def __init__(self, n_alpha: int, n_beta: int, spins: list, l_max: int,
                 arch: str | None = None, layer_style: str | None = None,
                 fft_method: str = "full", nonlinearity: str | None = None,
                 use_batch_norm: bool = True, use_layer_norm: bool = False,
                 fuse: bool = True,
                 wigner_n0: list | None = None, wigner_n1: list | None = None,
                 f_wigner: list | None = None):
        super().__init__()
        self.n_alpha, self.n_beta = n_alpha, n_beta
        #: Run the SO(3) chain slab by slab. Off restores the plain tensor path,
        #: which is what the fused one is checked against.
        self.fuse = fuse
        self.spec = resolve(arch, layer_style)
        self.arch = self.spec.name
        self.nonlinearity_type = self.spec.check_sigma(
            self.spec.sigma if nonlinearity is None else nonlinearity)
        self.use_batch_norm, self.use_layer_norm = use_batch_norm, use_layer_norm

        if fft_method != "full":
            raise ValueError(f"fft_method={fft_method!r}: only 'full' is implemented.")
        self.fft_method = fft_method

        self.spins = normalise_spins(spins)
        if len(self.spins[-1]) != 1:
            raise ValueError(f"Last spin must be a single value, got {self.spins[-1]}")
        self.output = "scalar" if self.spins[-1][0] == 0 else "vector"

        self.needs_0_spin = any(0 in s for s in self.spins)
        self.needs_1_spin = any(1 in s for s in self.spins)
        self.wigner_n0 = self._register("SLICE_WIGNER_N0", wigner_n0, l_max) if self.needs_0_spin else None
        self.wigner_n1 = self._register("SLICE_WIGNER_N1", wigner_n1, l_max) if self.needs_1_spin else None
        self.wigner_full = self._register("FULL_WIGNER", f_wigner, l_max) if self.spec.lifts else None

    #: Buffer name per table kind; the layout lives beside it, the bytes in the buffer.
    BUFFERS = {"SLICE_WIGNER_N0": "wigner_n0_packed",
               "SLICE_WIGNER_N1": "wigner_n1_packed",
               "FULL_WIGNER": "wigner_full_packed"}

    def _register(self, kind: str, table: list | None, l_max: int) -> precomp.Packed:
        """Pack one table into a single buffer, out of the state_dict."""
        table = precomp.load(kind, self.n_beta, l_max) if table is None else table
        packed = precomp.Packed.build(table)
        self.register_buffer(self.BUFFERS[kind], packed.packed, persistent=False)
        return packed

    def _tables(self, kind: str, layout) -> precomp.Packed | None:
        """The packed table on whatever device the module is on now."""
        if layout is None:
            return None
        return layout.on(self.get_buffer(self.BUFFERS[kind]))

    def get_buffered_slice_wigner(self, n: int = 1):
        """The spin-`n` column of the Wigner table on the current device, or None."""
        kind = "SLICE_WIGNER_N1" if n == 1 else "SLICE_WIGNER_N0"
        return self._tables(kind, self.wigner_n1 if n == 1 else self.wigner_n0)

    def get_buffered_full_wigner(self):
        return self._tables("FULL_WIGNER", self.wigner_full)

    def lifted(self, i: int) -> bool:
        """Does block `i` consume an SO(3) signal? Only scnn keeps one between blocks."""
        return i > 0 and self.spec.filters == "so3"

    def slices(self) -> dict:
        return dict(d_matrices_n0=self.get_buffered_slice_wigner(0),
                    d_matrices_n1=self.get_buffered_slice_wigner(1))

    def _io(self, L_in: int, L_out: int | None):
        """The input analysis, and (unless the network ends in a head) the output synthesis."""
        self.s2fft = S2FFT(L=L_in, n=self.spins[0])
        if L_out is not None:
            self.s2ifft = S2iFFT(L=L_out, n_alpha=self.n_alpha, n_beta=self.n_beta, n=self.spins[-1])
            if self.spec.lifts:
                self.so3_to_s2 = SO3_to_S2(L=L_out, n=self.spins[-1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, (C,) n_alpha, n_beta]` grid -> flat spectrum. Real input is read as `view_as_real`."""
        if x.dtype == torch.float32:
            x = torch.view_as_complex(x)
        if x.dim() < 4:
            x = x.reshape(x.size(0), 1, self.n_alpha, self.n_beta)
        return self.s2fft(x, **self.slices())

    def decode(self, x) -> torch.Tensor:
        """Last spectrum -> `[B, n_alpha, n_beta]` grid."""
        batch = x[0].size(0) if isinstance(x, list) else x.size(0)
        if self.spec.lifts:
            x = self.so3_to_s2(x)
        return self.s2ifft(x, **self.slices()).reshape(batch, self.n_alpha, self.n_beta)


class SO3CNN(EquivariantNetwork):
    """A flat stack at one band-limit: no pooling, no skips. The ablation baseline,
    and the only topology with an scnn path, since it never pools.

    `widths` has one entry per feature map, `spins` likewise; `resolution` is L.
    """

    def __init__(self, n_alpha: int, n_beta: int, resolution: int, widths: list, spins: list,
                 **kwargs):
        if resolution is None:
            raise ValueError("Resolution must be specified")
        if len(widths) != len(spins):
            raise ValueError("Widths and spins must have same length")
        if len(widths) < 2:
            raise ValueError("Must have at least 2 elements in widths (input and output)")
        super().__init__(n_alpha, n_beta, spins, resolution, **kwargs)
        self.resolutions, self.widths = resolution, widths

        n = len(widths) - 1
        self.layers = nn.ModuleList(
            Block(self, widths[i], widths[i + 1], self.spins[i + 1], resolution,
                  lifted_input=self.lifted(i), conv_only=i == n - 1)
            for i in range(n))
        self._io(resolution, resolution)

    def forward(self, x):
        x = self.encode(x)
        for block in self.layers:
            x = block(x, self)
        return self.decode(x)


class SO3UNet(EquivariantNetwork):
    """A UNet over a band-limit schedule, e.g. `resolutions=[179, 100, 50, 100, 179]`.

    Pooling truncates the spectrum and unpooling zero-pads it; each up block
    concatenates the matching down block's output (Ronneberger et al. 2015).
    `widths` has one more entry than `resolutions`.
    """

    def __init__(self, n_alpha: int, n_beta: int, resolutions: list, widths: list, spins: list,
                 **kwargs):
        if len(resolutions) != len(widths) - 1 or len(widths) != len(spins):
            raise ValueError("Widths and spins must have same length, and resolution must be one fewer element")
        if len(resolutions) < 2 or len(resolutions) % 2 != 1:
            raise ValueError("Layers must be at least two elements and be odd-numbered")
        super().__init__(n_alpha, n_beta, spins, max(resolutions), **kwargs)
        self.resolutions, self.widths = resolutions, widths

        n, half = len(resolutions), len(resolutions) // 2
        self.down = nn.ModuleList(
            Block(self, widths[i], widths[i + 1], self.spins[i + 1], resolutions[i],
                  lifted_input=self.lifted(i))
            for i in range(half + 1))
        self.up = nn.ModuleList(
            Block(self, widths[i] + widths[n - i], widths[i + 1], self.spins[i + 1], resolutions[i],
                  lifted_input=self.lifted(i), conv_only=i == n - 1)
            for i in range(half + 1, n))
        self._io(resolutions[0], resolutions[-1])

    def forward(self, x):
        x = self.encode(x)
        skips = []
        for i, block in enumerate(self.down):
            x = block(x, self)
            if i < len(self.down) - 1:
                skips.append(x)
                x = resize(x, self.resolutions[i + 1])
        for block, skip in zip(self.up, reversed(skips)):
            x = block(concatenate(skip, x), self)
        return self.decode(x)


class SO3Autoencoder(EquivariantNetwork):
    """`SO3UNet` without the skips: everything reaching the output passes the bottleneck."""

    def __init__(self, n_alpha: int, n_beta: int, resolutions: list, widths: list, spins: list,
                 **kwargs):
        if len(resolutions) != len(widths) - 1 or len(widths) != len(spins):
            raise ValueError("Widths and spins must have same length, and resolution must be one fewer element")
        if len(resolutions) < 2 or len(resolutions) % 2 == 0:
            raise ValueError("Layers must be at least two elements and be odd-numbered")
        super().__init__(n_alpha, n_beta, spins, max(resolutions), **kwargs)
        self.resolutions, self.widths = resolutions, widths

        n = len(resolutions)
        self.layers = nn.ModuleList(
            Block(self, widths[i], widths[i + 1], self.spins[i + 1], resolutions[i],
                  lifted_input=self.lifted(i), conv_only=i == n - 1)
            for i in range(n))
        self._io(resolutions[0], resolutions[-1])

    def forward(self, x):
        x = self.encode(x)
        for L, block in zip(self.resolutions, self.layers):
            x = block(resize(x, L), self)
        return self.decode(x)


class SO3MNISTClassifier(EquivariantNetwork):
    """An encoder down to L = 0, whose magnitude is rotation-invariant, then a linear head.

    Each block pools by analysing at the next entry of `resolutions`, which must
    end at 0; `widths[-1]` is the number of invariant features.
    """

    def __init__(self, n_alpha: int, n_beta: int, resolutions: list, widths: list, spins: list,
                 head: bool = True, n_classes: int = 10, **kwargs):
        super().__init__(n_alpha, n_beta, spins, max(resolutions), **kwargs)
        self.resolutions, self.widths, self.head = resolutions, widths, head

        self.layers = nn.ModuleList(
            Block(self, widths[i], widths[i + 1], self.spins[i + 1], resolutions[i],
                  L_out=resolutions[i + 1], lifted_input=self.lifted(i))
            for i in range(len(widths) - 1))
        self._io(resolutions[0], None)
        if head:
            self.final_layer = nn.Linear(widths[-1], n_classes)

    def forward(self, x):
        batch = x.size(0)
        x = self.encode(x)
        for block in self.layers:
            x = block(x, self)
        # On the group the l = 0 block is the invariant subspace; on S2 it is the
        # single remaining coefficient.
        if isinstance(x, list):
            x = x[0]
        if x.size(-1) != 1:
            raise ValueError(f"Final output should have only one order (L=0), got shape {tuple(x.shape)}")
        x = x.abs().reshape(batch, -1)
        return self.final_layer(x) if self.head else x
