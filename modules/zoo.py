"""One place where a model is described, for training and for testing alike.

`run_training` and `run_testing` each carried their own `if/elif` chain over
`(experiment, model_type)`, and the two had drifted: one omitted `head=True`,
one hardcoded a nonlinearity its architecture does not admit, and six branches
were spelled `mnist-...` where the validated name is `MNIST-...`, so they could
never run. A single table cannot drift from itself.

The entries are every pair the training chain could build -- the three
architectures on each of the seven experiments, plus the `Flat` and
`OnlyOneSpin` ablations -- taken from `run_training.py`, since that is what
produced the checkpoints on disk -- and Cohen's sCNN on the two MNIST
classification tasks, the only ones where it is cheap. Model types are named
after the architecture: `S3CNN*` is `s3cnn` (ours), `SwsCNN*` is `swscnn`
(Esteves et al.), `SCNN*` is `scnn` (Cohen et al.), `CNN*` is the planar
baseline; see `modules/architectures.py`.

Values a schedule sets are passed at build time and filtered against the
class's own signature, so a CNN baseline is not handed a spin argument. Where
an entry names one of those itself the entry wins, which is what the
`OnlyOneSpin` ablations relied on.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from modules.CNNAutoencoder import CNNAutoencoder512
from modules.CNNUNet import CNNUNet
from modules.MNIST_CNN import CNNMNISTClassifier
from modules.equivariant import SO3Autoencoder, SO3CNN, SO3MNISTClassifier, SO3UNet


@dataclass(frozen=True)
class Task:
    """What an experiment reads and predicts, independent of the model."""

    #: "weather" (ERA5, split by year) or "mnist" (Spherical MNIST).
    data: str
    #: Dataset-side field types: "scalar" | "vector", and "label" for `y`.
    x: str
    y: str
    n_alpha: int
    n_beta: int
    #: Weather data only.
    root: str | None = None
    #: The target is the input itself.
    autoencoding: bool = False

    @property
    def in_datatype(self) -> str:
        return "complex-" + self.x

    @property
    def out_datatype(self) -> str:
        return self.y if self.y == "label" else "complex-" + self.y

    @property
    def classification(self) -> bool:
        return self.y == "label"


_WIND = "dataset/wind->wind/weekly_coarse_torch/"
_TEMP = "dataset/temp->wind/weekly_coarse_torch/"

TASKS = {
    "wind->wind": Task("weather", "vector", "vector", 360, 179, root=_WIND),
    "temp->wind": Task("weather", "scalar", "vector", 360, 179, root=_TEMP),
    "autoencoder": Task("weather", "vector", "vector", 360, 179, root=_WIND,
                        autoencoding=True),
    "MNIST-scalar-classification": Task("mnist", "scalar", "label", 120, 59),
    "MNIST-vf-classification": Task("mnist", "vector", "label", 120, 59),
    "MNIST-scalar-to-vector": Task("mnist", "scalar", "vector", 120, 59),
    "MNIST-vector-to-scalar": Task("mnist", "vector", "scalar", 120, 59),
}


@dataclass(frozen=True)
class Spec:
    """How to build one (experiment, model_type), and where its results go."""

    cls: type
    #: Results land in `experiments/<experiment>/<mode>/`; it names the
    #: architecture, so a flat model and a UNet share a directory and are told
    #: apart by `log_id`.
    mode: str
    kwargs: dict = field(default_factory=dict)
    #: Rotate training data with the model's own Wigner buffers instead of a
    #: second copy. Set where the training chain set it: s3cnn on ERA5.
    share_d_matrices: bool = False


#: A band-limit L is transformed on a grid of 2(L+1) longitudes, and cuFFT is
#: fast only when that factors into small primes. **L = 100 gives 202 = 2 x 101**,
#: and 101 is prime, so those layers fall back to cuFFT's prime-size path: it
#: measured 73 ms of a 567 ms training step, about 60% of all the time spent in
#: transforms, for two layers out of five. L = 99 (200 = 2^3 x 5^2) or L = 101
#: (204 = 2^2 x 3 x 17) avoid it. The schedules below are left as they are
#: because they are what the published results were trained at; a new experiment
#: has no reason to pick 100.

ZOO = {

    # wind->wind -- ERA5 wind field one week ahead. The main result.
    ("wind->wind", "CNNUNet"): Spec(
        cls=CNNUNet, mode="CNN",
        kwargs=dict(n_alpha=360, n_beta=179, in_datatype='complex-vector',
            out_datatype='complex-vector')),
    ("wind->wind", "S3CNNUNet"): Spec(
        cls=SO3UNet, mode="S3CNN", share_d_matrices=True,
        kwargs=dict(arch='s3cnn', fft_method='full', n_alpha=360, n_beta=179,
            resolutions=[179, 100, 50, 100, 179],
            widths=[1, 2, 8, 30, 8, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),
    ("wind->wind", "SwsCNNUNet"): Spec(
        cls=SO3UNet, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=360, n_beta=179,
            resolutions=[179, 150, 100, 50, 100, 150, 179],
            widths=[1, 4, 32, 128, 256, 48, 16, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),
    # ablations
    ("wind->wind", "S3CNNFlat"): Spec(
        cls=SO3CNN, mode="S3CNN", share_d_matrices=True,
        kwargs=dict(arch='s3cnn', fft_method='full', n_alpha=360, n_beta=179,
            resolution=179, widths=[1, 2, 4, 2, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [1]])),
    ("wind->wind", "SwsCNNFlat"): Spec(
        cls=SO3CNN, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=360, n_beta=179, resolution=179,
            widths=[1, 4, 32, 128, 256, 48, 16, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),
    ("wind->wind", "S3CNNUNetOnlyOneSpin"): Spec(
        cls=SO3UNet, mode="S3CNN", share_d_matrices=True,
        kwargs=dict(arch='s3cnn', fft_method='full', n_alpha=360, n_beta=179,
            resolutions=[179, 100, 50, 100, 179],
            widths=[1, 2, 8, 30, 8, 1],
            spins=[[1], [1], [1], [1], [1], [1]])),
    ("wind->wind", "SwsCNNUNetOnlyOneSpin"): Spec(
        cls=SO3UNet, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=360, n_beta=179,
            resolutions=[179, 150, 100, 50, 100, 150, 179],
            widths=[1, 4, 32, 128, 256, 48, 16, 1],
            spins=[[1], [1], [1], [1], [1], [1], [1], [1]],
            nonlinearity='magnitude')),

    # temp->wind -- ERA5 temperature to wind field.
    ("temp->wind", "CNNUNet"): Spec(
        cls=CNNUNet, mode="CNN",
        kwargs=dict(n_alpha=360, n_beta=179, in_datatype='complex-scalar',
            out_datatype='complex-vector')),
    ("temp->wind", "S3CNNUNet"): Spec(
        cls=SO3UNet, mode="S3CNN", share_d_matrices=True,
        kwargs=dict(arch='s3cnn', fft_method='full', n_alpha=360, n_beta=179,
            resolutions=[179, 100, 50, 100, 179],
            widths=[1, 2, 8, 30, 8, 1],
            spins=[[0], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),
    ("temp->wind", "SwsCNNUNet"): Spec(
        cls=SO3UNet, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=360, n_beta=179,
            resolutions=[179, 150, 100, 50, 100, 150, 179],
            widths=[1, 4, 32, 128, 256, 48, 16, 1],
            spins=[[0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),
    # ablations
    ("temp->wind", "S3CNNFlat"): Spec(
        cls=SO3CNN, mode="S3CNN", share_d_matrices=True,
        kwargs=dict(arch='s3cnn', fft_method='full', n_alpha=360, n_beta=179,
            resolution=179, widths=[1, 1], spins=[[0], [1]])),
    ("temp->wind", "SwsCNNFlat"): Spec(
        cls=SO3CNN, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=360, n_beta=179, resolution=179,
            widths=[1, 4, 32, 128, 256, 48, 16, 1],
            spins=[[0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),

    # autoencoder -- Reconstruct the input field through a bottleneck.
    ("autoencoder", "CNNAutoencoder"): Spec(
        cls=CNNAutoencoder512, mode="CNN",
        kwargs=dict()),
    ("autoencoder", "S3CNNAutoencoder"): Spec(
        cls=SO3Autoencoder, mode="S3CNN", share_d_matrices=True,
        kwargs=dict(arch='s3cnn', n_alpha=360, n_beta=179,
            resolutions=[179, 100, 50, 5, 50, 100, 179],
            widths=[1, 2, 4, 7, 7, 4, 2, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),
    ("autoencoder", "SwsCNNAutoencoder"): Spec(
        cls=SO3Autoencoder, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=360, n_beta=179,
            resolutions=[179, 150, 100, 50, 5, 50, 100, 150, 179],
            widths=[1, 16, 24, 32, 7, 7, 32, 24, 16, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),

    # MNIST-scalar-classification -- Spherical MNIST, scalar input.
    ("MNIST-scalar-classification", "CNNMNIST"): Spec(
        cls=CNNMNISTClassifier, mode="CNN",
        kwargs=dict(in_datatype='complex-scalar', n_alpha=120, n_beta=59,
            widths=[1, 8, 32, 128, 256, 512])),
    ("MNIST-scalar-classification", "S3CNNMNIST"): Spec(
        cls=SO3MNISTClassifier, mode="S3CNN",
        kwargs=dict(n_alpha=120, n_beta=59, head=True, arch='s3cnn',
            fft_method='full', widths=[1, 2, 8, 32, 128, 512],
            resolutions=[59, 38, 14, 8, 2, 0],
            spins=[[0], [0, 1], [0, 1], [0, 1], [0, 1], [0]])),
    ("MNIST-scalar-classification", "SwsCNNMNIST"): Spec(
        cls=SO3MNISTClassifier, mode="SWSCNN",
        kwargs=dict(n_alpha=120, n_beta=59, head=True, arch='swscnn',
            fft_method='full', widths=[1, 4, 16, 32, 64, 128, 256, 512],
            resolutions=[59, 50, 40, 30, 20, 10, 2, 0],
            spins=[[0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0]])),
    # Cohen et al.'s sCNN, on S3CNNMNIST's band-limits. A group filter costs
    # sum_l (2l+1)^2 per channel pair against (L+1)^2, so the widths are cut to
    # match S3CNNMNIST's parameter count (~1.0M complex) rather than its widths.
    # The hidden spins only select the layer norm's mean subtraction: a signal
    # on the group has a mean in every channel.
    ("MNIST-scalar-classification", "SCNNMNIST"): Spec(
        cls=SO3MNISTClassifier, mode="SCNN",
        kwargs=dict(n_alpha=120, n_beta=59, head=True, arch='scnn',
            fft_method='full', widths=[1, 2, 2, 8, 24, 512],
            resolutions=[59, 38, 14, 8, 2, 0],
            spins=[[0], [0], [0], [0], [0], [0]])),
    # ablations
    ("MNIST-scalar-classification", "S3CNNMNISTOnlyOneSpin"): Spec(
        cls=SO3MNISTClassifier, mode="S3CNN",
        kwargs=dict(n_alpha=120, n_beta=59, head=True, arch='s3cnn',
            fft_method='full', widths=[1, 2, 8, 32, 128, 512],
            resolutions=[59, 38, 14, 8, 2, 0],
            spins=[[0], [0], [0], [0], [0], [0]])),
    ("MNIST-scalar-classification", "SwsCNNMNISTOnlyOneSpin"): Spec(
        cls=SO3MNISTClassifier, mode="SWSCNN",
        kwargs=dict(n_alpha=120, n_beta=59, head=True, arch='swscnn',
            fft_method='full', widths=[1, 4, 16, 32, 64, 128, 256, 512],
            resolutions=[59, 50, 40, 30, 20, 10, 2, 0],
            spins=[[0], [0], [0], [0], [0], [0], [0], [0]],
            nonlinearity='magnitude')),

    # MNIST-vf-classification -- Spherical MNIST, vector-field input.
    ("MNIST-vf-classification", "CNNMNIST"): Spec(
        cls=CNNMNISTClassifier, mode="CNN",
        kwargs=dict(in_datatype='complex-vector', n_alpha=120, n_beta=59,
            widths=[2, 8, 32, 128, 256, 512])),
    ("MNIST-vf-classification", "S3CNNMNIST"): Spec(
        cls=SO3MNISTClassifier, mode="S3CNN",
        kwargs=dict(n_alpha=120, n_beta=59, arch='s3cnn', fft_method='full',
            widths=[1, 2, 8, 32, 128, 512],
            resolutions=[59, 38, 14, 8, 2, 0],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0]])),
    # The scalar SwsCNNMNIST with a spin-1 input, as the vector S3CNN is its
    # scalar one. Until 2026-10-13 this was widths [1, 4, 16, 64, 256, 512] on
    # [59, 50, 25, 10, 2, 0], 1.3M real parameters against the others' 2.0M: the
    # vector S3CNN had been moved to its scalar configuration and this one never
    # was. Table 4's vector swsCNN row, and checkpoints from before that date,
    # are the old model and do not load into this one.
    ("MNIST-vf-classification", "SwsCNNMNIST"): Spec(
        cls=SO3MNISTClassifier, mode="SWSCNN",
        kwargs=dict(n_alpha=120, n_beta=59, arch='swscnn', fft_method='full',
            widths=[1, 4, 16, 32, 64, 128, 256, 512],
            resolutions=[59, 50, 40, 30, 20, 10, 2, 0],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0]])),
    ("MNIST-vf-classification", "SCNNMNIST"): Spec(
        cls=SO3MNISTClassifier, mode="SCNN",
        kwargs=dict(n_alpha=120, n_beta=59, arch='scnn', fft_method='full',
            widths=[1, 2, 2, 8, 24, 512],
            resolutions=[59, 38, 14, 8, 2, 0],
            spins=[[1], [0], [0], [0], [0], [0]])),

    # MNIST-scalar-to-vector -- Spherical MNIST to its Sobel gradient field.
    ("MNIST-scalar-to-vector", "CNNUNet"): Spec(
        cls=CNNUNet, mode="CNN",
        kwargs=dict(n_alpha=120, n_beta=59, in_datatype='complex-scalar',
            out_datatype='complex-vector')),
    ("MNIST-scalar-to-vector", "S3CNNUNet"): Spec(
        cls=SO3UNet, mode="S3CNN",
        kwargs=dict(arch='s3cnn', fft_method='full', n_alpha=120, n_beta=59,
            resolutions=[59, 50, 25, 50, 59],
            widths=[1, 2, 8, 30, 8, 1],
            spins=[[0], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),
    ("MNIST-scalar-to-vector", "SwsCNNUNet"): Spec(
        cls=SO3UNet, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=120, n_beta=59,
            resolutions=[59, 50, 40, 30, 40, 50, 59],
            widths=[1, 4, 32, 128, 256, 48, 16, 1],
            spins=[[0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])),

    # MNIST-vector-to-scalar -- the Sobel field back to the digit.
    ("MNIST-vector-to-scalar", "CNNUNet"): Spec(
        cls=CNNUNet, mode="CNN",
        kwargs=dict(n_alpha=120, n_beta=59, in_datatype='complex-vector',
            out_datatype='complex-scalar')),
    ("MNIST-vector-to-scalar", "S3CNNUNet"): Spec(
        cls=SO3UNet, mode="S3CNN",
        kwargs=dict(arch='s3cnn', fft_method='full', n_alpha=120, n_beta=59,
            resolutions=[59, 50, 25, 50, 59],
            widths=[1, 2, 8, 30, 8, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0]])),
    ("MNIST-vector-to-scalar", "SwsCNNUNet"): Spec(
        cls=SO3UNet, mode="SWSCNN",
        kwargs=dict(arch='swscnn', n_alpha=120, n_beta=59,
            resolutions=[59, 50, 40, 50, 40, 50, 59],
            widths=[1, 4, 32, 128, 256, 48, 16, 1],
            spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0]])),
}


def task(experiment: str) -> Task:
    """The experiment's data description, or a ValueError naming the known ones."""
    try:
        return TASKS[experiment]
    except KeyError:
        raise ValueError(f"Unknown experiment {experiment!r}; expected one of "
                         f"{sorted(TASKS)}") from None


def spec(experiment: str, model_type: str) -> Spec:
    """The entry for this pair, or a ValueError naming what is available."""
    task(experiment)
    try:
        return ZOO[(experiment, model_type)]
    except KeyError:
        known = sorted(m for e, m in ZOO if e == experiment)
        raise ValueError(f"{experiment!r} has no {model_type!r}; it has "
                         f"{known}") from None


def accepted(cls: type) -> set[str]:
    """Keywords `cls` takes, following `**kwargs` up to the base that names them."""
    names = set()
    for klass in cls.__mro__:
        params = inspect.signature(klass.__init__).parameters.values()
        names |= {p.name for p in params if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
        if not any(p.kind == p.VAR_KEYWORD for p in params):
            break
    return names - {"self"}


def build(experiment: str, model_type: str, **runtime):
    """The model for this pair, with the schedule's settings applied."""
    s = spec(experiment, model_type)
    kwargs = {k: v for k, v in runtime.items() if k in accepted(s.cls) and v is not None}
    kwargs.update(s.kwargs)          # an entry that names a value keeps it
    return s.cls(**kwargs)


def mode(experiment: str, model_type: str) -> str:
    """The results directory this pair writes into."""
    return spec(experiment, model_type).mode


MODEL_TYPES = sorted({m for _, m in ZOO})
EXPERIMENTS = sorted(TASKS)
