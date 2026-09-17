"""The model registry.

These are structural checks, not numerical ones: that every entry can actually
be called, that it names an architecture that exists, and that it does not ask
for a nonlinearity its architecture rejects. Each corresponds to a way the two
hand-written zoos had already gone wrong.
"""

from __future__ import annotations


import pytest

import run_testing
import run_training
from modules import zoo
from modules.architectures import ARCHITECTURES, resolve

KEYS = sorted(zoo.ZOO)


def test_the_registry_holds_every_pair_the_training_chain_could_build():
    assert len(zoo.ZOO) == 29
    assert zoo.EXPERIMENTS == ["MNIST-scalar-classification", "MNIST-scalar-to-vector",
                               "MNIST-vector-to-scalar", "MNIST-vf-classification",
                               "autoencoder", "temp->wind", "wind->wind"]
    assert {e for e, _ in zoo.ZOO} == set(zoo.TASKS)


@pytest.mark.parametrize("experiment", zoo.EXPERIMENTS)
def test_every_experiment_carries_all_three_architectures(experiment):
    """The point of the framework is one comparison across the same three
    architectures; an experiment missing a baseline cannot make it."""
    archs = {zoo.ZOO[k].kwargs.get("arch", "cnn") for k in zoo.ZOO if k[0] == experiment}
    assert archs == {"cnn", "s3cnn", "swscnn"}


@pytest.mark.parametrize("key", KEYS, ids=lambda k: f"{k[0]}:{k[1]}")
def test_every_entry_is_callable_as_written(key):
    """`head=True` reached one zoo and not the other; a keyword the class does
    not take would have been a TypeError at the first run of the experiment."""
    spec = zoo.ZOO[key]
    unknown = sorted(set(spec.kwargs) - zoo.accepted(spec.cls))
    assert not unknown, f"{spec.cls.__name__} does not take {unknown}"


@pytest.mark.parametrize("key", KEYS, ids=lambda k: f"{k[0]}:{k[1]}")
def test_no_entry_asks_for_a_nonlinearity_its_architecture_rejects(key):
    """One zoo hardcoded `magnitude` on an s3cnn model, which is spin-preserving
    only and not admissible there."""
    spec = zoo.ZOO[key]
    arch, sigma = spec.kwargs.get("arch"), spec.kwargs.get("nonlinearity")
    if arch is None or sigma is None:
        pytest.skip("a CNN baseline, or the nonlinearity comes from the schedule")
    resolve(arch).check_sigma(sigma)


@pytest.mark.parametrize("key", KEYS, ids=lambda k: f"{k[0]}:{k[1]}")
def test_every_entry_names_a_real_architecture(key):
    arch = zoo.ZOO[key].kwargs.get("arch")
    assert arch is None or arch in ARCHITECTURES


def test_the_mode_is_consistent_for_one_architecture():
    """`mode` is the results directory. Two entries with the same architecture
    wrote to different directories before: `temp->wind` put its flat S3CNN in
    `S3CNN/` and `wind->wind` put its in `S3CNN_FLAT/`."""
    seen = {}
    for key, spec in zoo.ZOO.items():
        arch = spec.kwargs.get("arch")
        if arch is None:
            continue
        seen.setdefault((arch, spec.cls.__name__), set()).add(spec.mode)
    mixed = {k: v for k, v in seen.items() if len(v) > 1}
    assert not mixed, f"same architecture, different results directories: {mixed}"


def test_an_unknown_pair_says_what_is_available():
    with pytest.raises(ValueError, match="has no"):
        zoo.spec("wind->wind", "NoSuchModel")
    with pytest.raises(ValueError, match="Unknown experiment"):
        zoo.spec("no-such-experiment", "S3CNNUNet")


@pytest.mark.parametrize("key", KEYS, ids=lambda k: f"{k[0]}:{k[1]}")
def test_every_entry_agrees_with_its_experiment_on_grid_and_datatypes(key):
    """The grid and field types are said twice, once per model and once per
    experiment; a UNet built for 360x179 fed 120x59 MNIST fails only at the
    first forward pass."""
    task, kwargs = zoo.TASKS[key[0]], zoo.ZOO[key].kwargs
    for name, want in [("n_alpha", task.n_alpha), ("n_beta", task.n_beta),
                       ("in_datatype", task.in_datatype),
                       ("out_datatype", task.out_datatype)]:
        if name in kwargs:
            assert kwargs[name] == want, f"{name}: {kwargs[name]!r} != {want!r}"


def test_wigner_sharing_is_only_asked_of_models_that_lift():
    """The trainer reads the rotation matrices off the model's own buffers."""
    for key, spec in zoo.ZOO.items():
        if spec.share_d_matrices:
            assert resolve(spec.kwargs["arch"]).lifts, key


def test_build_applies_the_schedule_but_an_entry_keeps_what_it_names(monkeypatch):
    class Model:
        def __init__(self, arch=None, nonlinearity=None, use_batch_norm=True):
            self.arch, self.nonlinearity, self.use_batch_norm = arch, nonlinearity, use_batch_norm

    monkeypatch.setitem(zoo.ZOO, ("wind->wind", "Fake"),
                        zoo.Spec(cls=Model, mode="S3CNN", kwargs=dict(nonlinearity="magnitude")))
    m = zoo.build("wind->wind", "Fake", nonlinearity="relu", use_batch_norm=False,
                  spins=[[1]], fft_method=None)
    assert (m.nonlinearity, m.use_batch_norm) == ("magnitude", False)


@pytest.mark.parametrize("key", [k for k in KEYS if zoo.ZOO[k].kwargs.get("arch")],
                         ids=lambda k: f"{k[0]}:{k[1]}")
def test_the_schedule_settings_reach_every_equivariant_model(key):
    """`build` filters by signature; the equivariant models take these through
    `**kwargs`, and a filter that stopped at the subclass dropped them silently."""
    assert {"nonlinearity", "use_batch_norm", "use_layer_norm", "fft_method"} <= \
        zoo.accepted(zoo.ZOO[key].cls)


@pytest.mark.parametrize("fn, extra", [
    (run_training.run_training, dict(total_epochs=1)),
    (run_testing.run_testing, dict(path="nowhere")),
], ids=["train", "test"])
def test_the_entry_points_reject_an_unknown_pair_before_touching_data(fn, extra):
    params = dict(operation="train", experiment="wind->wind", model_type="SO3UNet",
                  batch_size=1, rotate="no", **extra)
    with pytest.raises(ValueError, match="has no 'SO3UNet'"):
        fn(gpu_id=0, params=params)


def test_the_learning_rate_can_be_set_from_a_schedule():
    """It is untuned since the initialisation changed; a sweep needs it as a
    key, not an edit."""
    from utils.params import accepted
    assert "lr" in accepted(run_training.run_training)
