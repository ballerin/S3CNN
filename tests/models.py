"""Small assembled models, one per architecture and layer style.

Sizes are the smallest the constructors accept, so the suite runs in seconds.
Every model is built with `use_batch_norm=False`: batch norm is incompatible
with the SO(3) layer styles (defect 10), and pinning that separately keeps the
rest of the tests readable.
"""

from __future__ import annotations

import torch

from modules.MNIST_SO3 import SO3MNISTClassifier
from modules.SO3Autoencoder import SO3Autoencoder
from modules.SO3CNN import SO3CNN
from modules.SO3UNet import SO3UNet
from tests.conftest import L, N_ALPHA, N_BETA, tables

SEED = 0

# Inner resolution of the pooling architectures.  2*(2+1) = 6 divides
# N_ALPHA = 18, so a roll of the input grid is still a whole number of grid
# steps after pooling; see `test_models.test_pooling_needs_commensurate_grids`.
INNER = 2


def _common():
    n0, n1, full = tables(L, N_BETA)
    return dict(n_alpha=N_ALPHA, n_beta=N_BETA, use_batch_norm=False,
                wigner_n0=n0, wigner_n1=n1, f_wigner=full)


def _nl(style):
    """The SO(3) styles have no magnitude nonlinearity; they take a ReLU."""
    return {} if style == "spins" else dict(nonlinearity="leaky_relu")


def cnn(style="spins"):
    return SO3CNN(layer_style=style, resolution=L, widths=[1, 2, 1],
                  spins=[[0], [0, 1], [0]], **_nl(style), **_common())


def unet(style="spins", inner=INNER):
    return SO3UNet(layer_style=style, resolutions=[L, inner, L],
                   widths=[1, 2, 2, 1], spins=[[0], [0, 1], [0, 1], [0]],
                   **_nl(style), **_common())


def autoencoder(style="spins"):
    return SO3Autoencoder(layer_style=style, resolutions=[L, INNER, L],
                          widths=[1, 2, 2, 1], spins=[[0], [0, 1], [0, 1], [0]],
                          **_nl(style), **_common())


def mnist(style="spins"):
    return SO3MNISTClassifier(layer_style=style, resolutions=[L, INNER, 0],
                              widths=[1, 2, 10], spins=[[0], [0, 1], [0]],
                              **_nl(style), **_common())


BUILDERS = {
    "cnn_spins": lambda: cnn("spins"),
    "cnn_so3": lambda: cnn("so3"),
    "cnn_full_so3": lambda: cnn("full_so3"),
    "unet_spins": lambda: unet("spins"),
    "unet_so3": lambda: unet("so3"),
    "autoencoder_spins": lambda: autoencoder("spins"),
    "mnist_spins": lambda: mnist("spins"),
    "mnist_so3": lambda: mnist("so3"),
}

# Models that map the sphere to itself; the rest are classifiers.
GRID_MODELS = [k for k in BUILDERS if not k.startswith("mnist")]


def build(name: str):
    """A model in eval mode, from a fixed seed, so goldens are reproducible."""
    torch.manual_seed(SEED)
    return BUILDERS[name]().eval()


def sample_input(batch: int = 2, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, N_ALPHA, N_BETA, dtype=torch.complex64, generator=g)
