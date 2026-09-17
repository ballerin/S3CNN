"""Loading the precomputed Wigner tables from `precomp/`.

The tables depend on both the latitude count and the maximum order, so the
filename carries both. Files written before that convention are named by
`n_beta` alone and are still accepted, since regenerating them takes hours.
"""

from __future__ import annotations

import os
import pickle

import torch

DIR = "precomp"


def path(kind: str, n_beta: int, l_max: int | None = None) -> str:
    """The file to read, preferring the two-part name when it exists."""
    if l_max is not None:
        both = os.path.join(DIR, f"{kind}_{n_beta}_{l_max}.list")
        if os.path.exists(both):
            return both
    return os.path.join(DIR, f"{kind}_{n_beta}.list")


def as_real(tables: list) -> list:
    """The tables as real fp32. `d^l_{mn}(beta)` is real by construction, and
    storing it complex doubles both the bytes and the multiply count."""
    out = []
    for m in tables:
        if m.is_complex():
            if torch.any(m.imag != 0):
                raise ValueError("Wigner table has a non-zero imaginary part; "
                                 "it is real by construction, so this is a generation bug.")
            m = m.real
        out.append(m.to(torch.float32).contiguous().requires_grad_(False))
    return out


def load(kind: str, n_beta: int, l_max: int | None = None) -> list:
    """A table as a list of real tensors, detached from the autograd graph."""
    with open(path(kind, n_beta, l_max), "rb") as fh:
        return as_real(pickle.load(fh))


class Packed:
    """A list of per-order Wigner tables held as one tensor.

    The models used to register `L+1` buffers per table and rebuild a Python
    list of them, by string-formatted lookup, on every call -- roughly four
    times per layer. Here the orders live in one contiguous buffer that moves
    with the module, and `tables[l]` is a view into it.

    `torch.nested` was the previous answer for the ragged full table; it is a
    prototype API that warns on construction, indexes slowly and blocks
    `torch.compile`.
    """

    def __init__(self, packed: torch.Tensor, shapes: list, cache: dict | None = None):
        self.packed = packed
        self.shapes = shapes
        #: Derived layouts (the dense (m, l) blocks), keyed by what they depend on.
        self.cache = {} if cache is None else cache
        self.offsets = [0]
        for shape in shapes:
            n = 1
            for d in shape[1:]:
                n *= d
            self.offsets.append(self.offsets[-1] + n)

    @classmethod
    def build(cls, tables: list) -> "Packed":
        """Pack real tables that share their first (latitude) axis."""
        tables = as_real(tables)
        n_beta = tables[0].shape[0]
        flat = torch.cat([t.reshape(n_beta, -1) for t in tables], dim=1)
        return cls(flat, [tuple(t.shape) for t in tables])

    def on(self, packed: torch.Tensor) -> "Packed":
        """The same layout over another copy of the buffer, e.g. after `.to()`."""
        return Packed(packed, self.shapes, cache=self.cache)

    def __len__(self) -> int:
        return len(self.shapes)

    def __getitem__(self, l: int) -> torch.Tensor:
        shape = self.shapes[l]
        return self.packed[:, self.offsets[l]:self.offsets[l + 1]].view(shape)

    def __iter__(self):
        return (self[l] for l in range(len(self)))


def rectangular(tables, L: int, n: int) -> torch.Tensor:
    """The per-order tables as one dense `[2L+1, L+1, n_beta]` block.

    Order `l` of a spin-`n` table is a column `[n_beta, 2l+1]`; laying them out
    by `(m, l)` and zero-padding where `l < |m|` turns the per-order loop in the
    S2 transforms into one batched matmul. The padding is what it costs: the
    dense block is about twice the triangular data.

    Spin 1 has no `l = 0` term, so its list starts at `l = 1` (see
    CONVENTIONS.md) and the first column here stays zero.
    """
    first = tables[0]
    n_beta = first.shape[0]
    out = torch.zeros((2 * L + 1, L + 1, n_beta), dtype=first.dtype, device=first.device)
    for l in range(1 if n == 1 else 0, L + 1):
        table = tables[l - 1 if n == 1 else l]          # [n_beta, 2l+1]
        out[L - l:L + l + 1, l, :] = table.transpose(0, 1)
    return out


def flat_index(L: int, device=None) -> torch.Tensor:
    """Where each `(m, l)` cell lands in the flat layout, and vice versa.

    Flat order `l` occupies `l**2 : (l+1)**2` for both spins, so one index
    serves both: cell `(m, l)` is flat position `l**2 + (m + l)`.
    """
    m = torch.arange(-L, L + 1, device=device).reshape(-1, 1)
    l = torch.arange(0, L + 1, device=device).reshape(1, -1)
    inside = m.abs() <= l
    return (l * l + m + l).clamp_min(0), inside


class Rect:
    """The index that maps the `(m, l)` layout to the flat one and back."""

    def __init__(self, like: torch.Tensor, L: int):
        self.L = L
        self.index, self.inside = flat_index(L, device=like.device)

    def to_flat(self, x: torch.Tensor) -> torch.Tensor:
        """`[..., 2L+1, L+1]` -> `[..., (L+1)**2]`."""
        flat = torch.zeros((*x.shape[:-2], (self.L + 1) ** 2),
                           dtype=x.dtype, device=x.device)
        idx = self.index[self.inside].reshape(-1)
        return flat.index_copy_(-1, idx, x[..., self.inside].reshape(*x.shape[:-2], -1))

    def from_flat(self, x: torch.Tensor) -> torch.Tensor:
        """`[..., (L+1)**2]` -> `[..., 2L+1, L+1]`, zero where `l < |m|`."""
        gathered = x[..., self.index.reshape(-1)].reshape(*x.shape[:-1], 2 * self.L + 1, self.L + 1)
        return gathered * self.inside


def _cached(tables, key, build):
    """Memoise a derived layout on the `Packed` that owns the bytes.

    A plain list of tables has nowhere to keep it, so it rebuilds; that is the
    test path, not the training one.
    """
    cache = getattr(tables, "cache", None)
    if cache is None:
        return build()
    if key not in cache:
        cache[key] = build()
    return cache[key]


def synthesis_table(tables, L: int, n: int) -> torch.Tensor:
    """The dense `(m, l)` block used to go from coefficients back to latitudes."""
    device = tables[0].device
    return _cached(tables, ("synthesis", L, n, str(device)),
                   lambda: rectangular(tables, L, n))


def analysis_table(rule, tables, L: int, n: int, n_beta: int) -> torch.Tensor:
    """The dense `(m, l)` block with the quadrature folded in.

    Folding the weights in once, rather than on every call, is what makes the
    batched contraction worth it: at L = 179 this block is 46 MB and touching
    it per call would cost more than the matmul.
    """
    device = tables[0].device

    def build():
        d = synthesis_table(tables, L, n)
        w = rule.node_weights(n_beta, device=device, dtype=torch.float32).to(d.dtype)
        scale = torch.tensor([rule.measure(n_beta) * rule.order_scale(l) for l in range(L + 1)],
                             dtype=d.dtype, device=device)
        return d * scale.reshape(1, -1, 1) * w.reshape(1, 1, -1)

    return _cached(tables, ("analysis", rule.name, L, n, n_beta, str(device)), build)


def rect_index(L: int, device=None) -> "Rect":
    """The flat <-> (m, l) index for band-limit L, built once per device."""
    key = (L, str(device))
    if key not in _INDEX:
        _INDEX[key] = Rect(torch.zeros(1, device=device), L)
    return _INDEX[key]


_INDEX: dict = {}


_ORDER_WEIGHTS: dict = {}


def order_weights(total: int, device=None) -> torch.Tensor:
    """`1/(2l+1)` for every coefficient of a packed SO(3) spectrum, zero on l = 0.

    The weight a normalisation gives each order, as one vector over the packed
    buffer, so the sum over orders is a single reduction rather than a loop.
    Order l = 0 is excluded: it carries the mean, not a deviation from it.
    """
    key = (total, str(device))
    if key not in _ORDER_WEIGHTS:
        w, l, at = [], 0, 0
        while at < total:
            size = (2 * l + 1) ** 2
            w.append(torch.full((size,), 0.0 if l == 0 else 1.0 / (2 * l + 1)))
            at += size
            l += 1
        _ORDER_WEIGHTS[key] = torch.cat(w)[:total].to(device)
    return _ORDER_WEIGHTS[key]
