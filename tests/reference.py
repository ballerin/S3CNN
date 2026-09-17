"""Independent float64 reference transforms, used to test the fast ones.

Nothing here shares code with `modules/`: the Wigner matrices come from the
explicit factorial formula rather than a matrix exponential, and the transforms
are direct sums rather than FFTs.  Slow by design.

Index conventions, verified against `modules.SO3lib` in `test_wigner.py`:
  * `wigner_d(l, betas)[b, i, j]` holds d^l_{m,n}(beta_b) with m = i - l
    and n = j - l (Varshalovich et al. 1988, eq. 4.3.1);
  * spectral coefficients are flat, order l occupying `l**2 : (l+1)**2`;
  * spin-1 signals reuse the spin-0 layout with a structural zero at index 0.

The factorial formula cancels catastrophically above about l = 30, so it is
only a reference at low order; the suite stays at l <= 16.
"""

from __future__ import annotations

import math

import numpy as np


def wigner_d(l: int, betas: np.ndarray) -> np.ndarray:
    """Wigner (small) d-matrices, shape [len(betas), 2l+1, 2l+1]."""
    betas = np.asarray(betas, dtype=np.float64)
    c, s = np.cos(betas / 2), np.sin(betas / 2)
    out = np.zeros((betas.size, 2 * l + 1, 2 * l + 1))
    lg = math.lgamma
    for i in range(2 * l + 1):
        mp = i - l
        for j in range(2 * l + 1):
            m = j - l
            norm = 0.5 * (lg(l + mp + 1) + lg(l - mp + 1)
                          + lg(l + m + 1) + lg(l - m + 1))
            acc = np.zeros_like(betas)
            for k in range(max(0, m - mp), min(l + m, l - mp) + 1):
                logw = norm - (lg(l + m - k + 1) + lg(k + 1)
                               + lg(mp - m + k + 1) + lg(l - mp - k + 1))
                acc += ((-1.0) ** (mp - m + k) * math.exp(logw)
                        * c ** (2 * l + m - mp - 2 * k) * s ** (mp - m + 2 * k))
            out[:, i, j] = acc
    return out


def wigner_column(l: int, betas: np.ndarray, n: int) -> np.ndarray:
    """The spin-n slice the S2 transforms use, shape [len(betas), 2l+1]."""
    return wigner_d(l, betas)[:, :, l + n]


def grid(n_alpha: int, n_beta: int) -> tuple[np.ndarray, np.ndarray]:
    """The equirectangular sampling grid; beta excludes both poles."""
    return (2 * np.pi * np.arange(n_alpha) / n_alpha,
            np.pi * (np.arange(n_beta) + 1) / (n_beta + 1))


def synthesize_s2(f_hat: np.ndarray, n: int, n_alpha: int,
                  betas: np.ndarray) -> np.ndarray:
    """Exact synthesis, [..., (L+1)**2] -> [..., n_alpha, n_beta].

    f(a_j, b) = n_alpha**-0.5 * sum_{l,m} f_hat[l,m] d^l_{m,n}(b) exp(-i m a_j)
    """
    L = int(round(math.sqrt(f_hat.shape[-1]))) - 1
    alphas, _ = grid(n_alpha, 1)
    out = np.zeros(f_hat.shape[:-1] + (n_alpha, betas.size), dtype=np.complex128)
    for l in range(abs(n), L + 1):
        d = wigner_column(l, betas, n)                       # [beta, m]
        coef = f_hat[..., l * l:(l + 1) * (l + 1)]           # [..., m]
        phase = np.exp(-1j * np.outer(alphas, np.arange(-l, l + 1)))  # [a, m]
        out += np.einsum("...m,bm,am->...ab", coef, d, phase)
    return out / math.sqrt(n_alpha)


def analyze_s2(f_fn, n: int, L: int, n_alpha: int, n_quad: int | None = None):
    """Ideal L2 projection of a callable, using Gauss-Legendre in cos(beta).

    `f_fn(alphas, betas)` must return an array of shape [n_alpha, n_beta].
    Exact for band-limited inputs, so it doubles as the accuracy target for
    the quadrature used in `modules.SO3fft`.
    """
    n_quad = n_quad or 4 * (L + 1)
    x, w = np.polynomial.legendre.leggauss(n_quad)
    betas = np.arccos(x)
    alphas, _ = grid(n_alpha, 1)
    f = np.asarray(f_fn(alphas, betas)) * math.sqrt(n_alpha)

    out = np.zeros((L + 1) ** 2, dtype=np.complex128)
    for l in range(abs(n), L + 1):
        d = wigner_column(l, betas, n)                       # [beta, m]
        phase = np.exp(1j * np.outer(alphas, np.arange(-l, l + 1)))
        g = np.einsum("ab,am->bm", f, phase) / n_alpha       # [beta, m]
        out[l * l:(l + 1) * (l + 1)] = (2 * l + 1) / 2 * np.einsum("b,bm,bm->m", w, d, g)
    return out


def wigner_D(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """D^l_{m,s}(Z(a)Y(b)Z(c)) = exp(-i m a) d^l_{m,s}(b) exp(-i s c)."""
    m = np.arange(-l, l + 1)
    return (np.exp(-1j * m * alpha)[:, None] * wigner_d(l, np.array([beta]))[0]
            * np.exp(-1j * m * gamma)[None, :])


def rotate_spectrum(f_hat: np.ndarray, alpha: float, beta: float,
                    gamma: float) -> np.ndarray:
    """Left action of Z(a)Y(b)Z(c) on flat spectral coefficients.

    Writing x = sum f_hat[l,m] D^l_{m,n}, the action x -> x(B^-1 . ) sends
    f_hat^l to conj(D^l(B)) f_hat^l.  Verified against an alpha-roll in
    `test_equivariance.py`.
    """
    L = int(round(math.sqrt(f_hat.shape[-1]))) - 1
    out = np.empty_like(f_hat, dtype=np.complex128)
    for l in range(L + 1):
        sl = slice(l * l, (l + 1) * (l + 1))
        out[..., sl] = np.einsum("ms,...s->...m",
                                 np.conj(wigner_D(l, alpha, beta, gamma)),
                                 f_hat[..., sl])
    return out
