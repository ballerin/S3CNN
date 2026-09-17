"""The quadrature over beta, as a swappable rule.

Analysis integrates a signal against `d^l_{mn}(beta)` over the latitudes. The
rule below is the one the package has always used -- a sine-weighted rectangle
rule on the equiangular nodes -- and it is not exact at the band-limit the
models are configured with: the roundtrip loses 21-33% at `L = n_beta`. Three
strict xfails in `tests/test_transforms.py` hold that open.

This module exists so that replacing the rule is a swap rather than surgery. A
rule owns the nodes and the complete analysis table

    A^l_{b,m} = measure * order_scale(l) * node_weights(b) * d^l_{b,m}

Candidates differ in how much of that they keep:

- **pseudo-inverse analysis** replaces the whole table with `pinv(D)` per order,
  which is not diagonal in `b` -- hence `operator()` rather than a weight vector;
- **beta-oversampling** keeps this rule and raises `n_beta` above `L + 1`;
- **Gauss-Legendre** also moves the nodes, so it needs the data resampled off
  the equirectangular ERA5 grid. The seam does not make that part free.

A rule that factorises says so with `diagonal = True`, and the transforms then
contract the pieces without ever materialising `A`. That matters on the SO(3)
path, where `A` is the size of the whole Wigner table.
"""

from __future__ import annotations

import torch

from .SO3lib import from_index_to_beta


class Quadrature:
    """A rule for integrating over beta. See `RectangleRule` for the shipped one."""

    name = "abstract"
    #: True when the table factorises as measure * order_scale * node_weights * d.
    diagonal = False

    def betas(self, n_beta: int, device=None) -> torch.Tensor:
        raise NotImplementedError

    def measure(self, n_beta: int) -> float:
        """The node spacing; a constant for an equiangular rule."""
        raise NotImplementedError

    def order_scale(self, l: int) -> float:
        """The per-order factor of the inversion formula."""
        raise NotImplementedError

    def node_weights(self, n_beta: int, device=None, dtype=torch.complex64) -> torch.Tensor:
        """The per-latitude factor. Only meaningful when `diagonal`."""
        raise NotImplementedError

    def operator(self, d: torch.Tensor, l: int, n_beta: int) -> torch.Tensor:
        """The complete analysis table for order `l`, shaped like `d^l` itself.

        `d` is the Wigner table for this order -- a spin column `[n_beta, 2l+1]`
        on S2, the full `[n_beta, 2l+1, 2l+1]` block on SO(3). The general path
        contracts the result directly, so a non-diagonal rule implements this
        one method and nothing else.
        """
        w = self.node_weights(n_beta, device=d.device, dtype=d.dtype)
        scale = self.measure(n_beta) * self.order_scale(l)
        return scale * w.reshape(-1, *([1] * (d.dim() - 1))) * d


class RectangleRule(Quadrature):
    """Sine-weighted rectangle rule on `beta_j = pi (j+1) / (n_beta+1)`.

    Exact for nothing in particular, and rank-deficient at `L = n_beta`, where
    there are more unknowns than samples. See `CONVENTIONS.md`.

    Candidates for replacing it: a pseudo-inverse analysis operator, Gauss-
    Legendre nodes, or oversampling beta.
    """

    name = "rectangle"
    diagonal = True

    def __init__(self):
        #: sin(beta) per (n_beta, device, dtype); it was rebuilt on every call.
        self._weights: dict = {}

    def betas(self, n_beta: int, device=None) -> torch.Tensor:
        return from_index_to_beta(torch.arange(0, n_beta, device=device), n_beta=n_beta)

    def measure(self, n_beta: int) -> float:
        return torch.pi / (n_beta + 1)  # n_beta + 1 intervals

    def order_scale(self, l: int) -> float:
        return (2 * l + 1) / 2

    def node_weights(self, n_beta: int, device=None, dtype=torch.float32) -> torch.Tensor:
        key = (n_beta, str(device), dtype)
        if key not in self._weights:
            self._weights[key] = torch.sin(self.betas(n_beta, device=device).to(dtype))
        return self._weights[key]


RULES = {rule.name: rule for rule in [RectangleRule()]}

DEFAULT = "rectangle"


def resolve(quadrature: str | Quadrature | None = None) -> Quadrature:
    """The rule named, or the default one."""
    if quadrature is None:
        return RULES[DEFAULT]
    if isinstance(quadrature, Quadrature):
        return quadrature
    try:
        return RULES[quadrature]
    except KeyError:
        raise ValueError(f"Unknown quadrature {quadrature!r}; expected one of "
                         f"{sorted(RULES)}") from None
