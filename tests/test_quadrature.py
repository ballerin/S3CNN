"""The quadrature seam.

The shipped rule is not accurate at the band-limit the models use, which the
strict xfails in `test_transforms.py` hold open. What is pinned here is the
seam itself: that the default rule is exactly the arithmetic the transforms
used before it existed, and that another rule can be dropped in without
touching them.
"""

from __future__ import annotations

import pytest
import torch

from modules import quadrature
from modules.SO3fft import S2FFT, SO3FFT
from tests.conftest import L, N_ALPHA, N_BETA, relerr, tables

RULE = quadrature.RULES["rectangle"]


def test_the_default_is_the_rule_the_package_shipped_with():
    assert quadrature.resolve() is RULE
    assert quadrature.resolve("rectangle") is RULE
    assert RULE.diagonal


def test_an_unknown_rule_says_what_is_available():
    with pytest.raises(ValueError, match="Unknown quadrature"):
        quadrature.resolve("gauss_legendre")


def test_the_nodes_are_the_equiangular_ones_excluding_the_poles():
    betas = RULE.betas(N_BETA)
    assert betas.shape == (N_BETA,)
    assert 0 < betas[0] < betas[-1] < torch.pi
    assert torch.allclose(betas[1:] - betas[:-1],
                          torch.full((N_BETA - 1,), RULE.measure(N_BETA)))


def test_the_factorised_pieces_multiply_out_to_the_operator():
    """The diagonal fast path avoids building this table; it must still be the
    table it is avoiding. On SO(3) it is the whole Wigner table, so the general
    path is only for rules that cannot factorise."""
    _, _, full = tables(L, N_BETA)
    for l in (0, 1, L):
        got = RULE.operator(full[l], l, N_BETA)
        w = RULE.node_weights(N_BETA).reshape(-1, 1, 1)
        want = RULE.measure(N_BETA) * RULE.order_scale(l) * w * full[l]
        assert relerr(got.numpy(), want.numpy()) < 1e-6


class Doubled(quadrature.Quadrature):
    """A non-diagonal stand-in: the shipped table, twice."""

    name = "doubled"
    diagonal = False

    def operator(self, d, l, n_beta):
        return 2 * RULE.operator(d, l, n_beta)


@pytest.mark.parametrize("n", [0, 1])
def test_a_rule_that_cannot_factorise_still_plugs_into_s2_analysis(n):
    """A pseudo-inverse analysis operator is not diagonal in beta, so it
    arrives as a whole table and the transform contracts it directly."""
    n0, n1, _ = tables(L, N_BETA)
    f = torch.randn(2, 1, N_ALPHA, N_BETA, dtype=torch.complex64)
    shipped = S2FFT(L=L, n=n)(f, n0, n1)
    doubled = S2FFT(L=L, n=n, quadrature=Doubled())(f, n0, n1)
    assert relerr(doubled.numpy(), 2 * shipped.numpy()) < 1e-6


def test_a_rule_that_cannot_factorise_still_plugs_into_so3_analysis():
    _, _, full = tables(L, N_BETA)
    x = torch.randn(1, 1, 2 * (L + 1), N_BETA, 2 * (L + 1), dtype=torch.complex64)
    shipped = SO3FFT(L=L)(x, d_matrices=full)
    doubled = SO3FFT(L=L, quadrature=Doubled())(x, d_matrices=full)
    for a, b in zip(doubled, shipped):
        assert relerr(a.numpy(), 2 * b.numpy()) < 1e-6


def test_naming_the_default_rule_changes_nothing():
    n0, n1, _ = tables(L, N_BETA)
    f = torch.randn(2, 1, N_ALPHA, N_BETA, dtype=torch.complex64)
    assert torch.equal(S2FFT(L=L, n=0)(f, n0, n1),
                       S2FFT(L=L, n=0, quadrature="rectangle")(f, n0, n1))
