"""The three architectures as a table rather than as branches.

What is pinned here is the vocabulary: which architecture each preset names,
which nonlinearities it admits, and what the pre-M2 `layer_style` strings map
onto.  The last of those is worth a test of its own -- `full_so3` was the sCNN
path and reads like it ought to be ours.
"""

from __future__ import annotations

import pytest
import torch

from modules.architectures import ARCHITECTURES, LEGACY_LAYER_STYLES, resolve
from modules.equivariant import SO3Autoencoder, SO3CNN, SO3MNISTClassifier, SO3UNet
from tests import models


# ------------------------------------------------------------------ the table

@pytest.mark.parametrize("arch, filters, projects, lifts, sigma", [
    ("scnn",   "so3",    False, True,  "leaky_relu"),
    ("s3cnn",  "spin",   True,  True,  "leaky_relu"),
    ("swscnn", "degree", False, False, "magnitude"),
])
def test_the_presets_are_what_the_paper_says(arch, filters, projects, lifts, sigma):
    spec = resolve(arch)
    assert (spec.filters, spec.projects, spec.lifts, spec.sigma) == \
        (filters, projects, lifts, sigma)


@pytest.mark.parametrize("legacy, arch", [
    ("full_so3", "scnn"),     # Cohen's group filter, not ours
    ("so3", "s3cnn"),         # ours
    ("spins", "swscnn"),      # Esteves'
])
def test_the_old_layer_style_names_still_resolve(legacy, arch):
    assert resolve(layer_style=legacy).name == arch
    assert LEGACY_LAYER_STYLES[legacy] == arch


def test_an_architecture_is_named_one_way_or_the_other():
    with pytest.raises(ValueError, match="not both"):
        resolve(arch="scnn", layer_style="full_so3")
    with pytest.raises(ValueError, match="required"):
        resolve()


@pytest.mark.parametrize("bad", ["so3cnn", "spins", "so3", "full_so3", ""])
def test_resolve_rejects_an_unknown_architecture(bad):
    """The legacy spellings are not architecture names; they go to
    `layer_style`, which is where the translation is documented."""
    with pytest.raises(ValueError, match="Unknown arch"):
        resolve(arch=bad)


# ----------------------------------------------------- admissible nonlinearities

SIGMAS = ["leaky_relu", "relu", "magnitude", "phase_collapse"]


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
@pytest.mark.parametrize("sigma", SIGMAS)
def test_a_model_takes_exactly_the_nonlinearities_its_architecture_admits(arch, sigma):
    """Before M2 the six inadmissible pairs built a model with an empty
    `nonlinearities` list and raised `IndexError` inside `forward`."""
    spec = resolve(arch)
    build = lambda: SO3CNN(arch=arch, nonlinearity=sigma, resolution=models.L,
                           widths=[1, 2, 1], spins=[[0], [0, 1], [0]],
                           **models._common())
    if sigma in spec.sigmas:
        build()(models.sample_input())
    else:
        with pytest.raises(ValueError, match="admits"):
            build()


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_the_default_nonlinearity_is_the_architectures_own(arch):
    model = SO3CNN(arch=arch, resolution=models.L, widths=[1, 2, 1],
                   spins=[[0], [0, 1], [0]], **models._common())
    assert model.nonlinearity_type == resolve(arch).sigma


# ------------------------------------------------------- pooling on the group

@pytest.mark.parametrize("cls, kwargs", [
    (SO3UNet, dict(resolutions=[models.L, models.INNER, models.L],
                   widths=[1, 2, 2, 1], spins=[[0], [0, 1], [0, 1], [0]])),
    (SO3Autoencoder, dict(resolutions=[models.L, models.INNER, models.L],
                          widths=[1, 2, 2, 1], spins=[[0], [0, 1], [0, 1], [0]])),
    (SO3MNISTClassifier, dict(resolutions=[models.L, models.INNER, 0],
                              widths=[1, 2, 10], spins=[[0], [0, 1], [0]])),
])
def test_every_topology_pools_an_scnn_on_the_group(cls, kwargs):
    """The signal stays on SO(3) between blocks, so pooling drops whole orders
    rather than truncating a flat S2 spectrum."""
    torch.manual_seed(models.SEED)
    model = cls(arch="scnn", **kwargs, **models._common()).eval()
    out = model(models.sample_input())
    out = torch.view_as_real(out) if out.is_complex() else out
    assert torch.isfinite(out).all()
