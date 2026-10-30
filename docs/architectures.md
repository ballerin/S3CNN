# Architectures and models

Three architectures, four topologies, and one table that pairs them with the
experiments. `modules/architectures.py` holds the architectures as frozen
dataclasses, so they are configuration rather than code paths.

## The three architectures

| `arch` | equivariance imposed | filter | $\sigma$ admitted | after |
| ------ | -------------------- | ------ | ----------------- | ----- |
| `scnn` | at the output | $\psi^l_{m,n}$, $O(L^3)$ | unconstrained | Cohen et al. 2018 |
| `s3cnn` | after every $\sigma$ | $\psi^l_n$, $O(L^2)$ | unconstrained | **ours** |
| `swscnn` | inside the filter | $\psi^l$, $O(L)$ | spin-preserving | Esteves et al. 2020, 2023 |

`scnn` and `s3cnn` lift to $SO(3)$; `s3cnn` projects back to $S^2$ after every
nonlinearity with the smoothing operator, `scnn` stays on the group until the
output. `swscnn` never leaves $S^2$. For an input of spin $p$ the filter
coefficients $\psi^l_{s,n}$ contribute only at $s = p$, so **the first layer of
an `scnn` and of an `s3cnn` are the same operator** — they diverge from layer
two onward.

`resolve(arch=..., layer_style=...)` also accepts the pre-M2 spellings, which
are easy to get backwards: `full_so3` was `scnn` (Cohen's), `so3` was `s3cnn`
(ours), `spins` was `swscnn`.

### Nonlinearities

`leaky_relu` (default for `scnn`, `s3cnn`) and `relu` act on real and imaginary
parts after a learned complex affine map. `magnitude` (default for `swscnn`)
acts on the magnitude and leaves the phase alone, which is what preserves the
spin; spin-0 channels carry no phase, so they take a leaky ReLU instead.
`phase_collapse` concatenates $|x|$ to $x$ and projects back (after Guth et
al.). Passing one an architecture does not admit raises, naming the reason.

## Topologies

All four are stacks of the same `Block` (`modules/equivariant.py`) and differ
only in wiring. `widths` and `spins` have one entry per feature map;
`spins` entries are `[0]`, `[1]` or `[0, 1]` (half the channels each), and the
last must be a single value.

| class | shape | notes |
| ----- | ----- | ----- |
| `SO3CNN` | flat stack at one band-limit | no pooling, no skips |
| `SO3UNet` | band-limit schedule, e.g. `[179, 100, 50, 100, 179]` | skips concatenate along channels; `resolutions` must be odd-length |
| `SO3Autoencoder` | the same schedule without skips | bottleneck at the smallest band-limit |
| `SO3MNISTClassifier` | schedule ending at $L = 0$ | `head=True` adds the classifier head |

Pooling is **truncation of the spectrum**, unpooling is zero-padding
(`equivariant.resize`); Cohen et al. instead max-pool on the group grid.

Planar baselines — `CNNUNet`, `CNNAutoencoder512`, `CNNMNISTClassifier` — take
`in_datatype`/`out_datatype` and `widths` instead, and are not equivariant.

### Normalisation

`use_batch_norm` and `use_layer_norm` select a spectral normalisation matched
to the domain: `FullSpectral*` on the group path, `SpinSpectral*` on $S^2$.
Layer norm on the fused path keeps the orders packed and costs one reduction
and one divide; batch norm forces them apart and costs a block per order.

## The model zoo

`modules/zoo.py` is the one place a model is described; `run_training` and
`run_testing` both read it, so they cannot drift. It pairs an experiment with a
`model_type`:

```python
from modules import zoo
model = zoo.build("wind->wind", "S3CNNUNet")
```

`TASKS` says what an experiment reads and predicts (`data`, field types,
$n_\alpha$, $n_\beta$, root); `ZOO` maps `(experiment, model_type)` to a class
and its kwargs. Model types are named after the architecture: `S3CNN*` is
`s3cnn`, `SwsCNN*` is `swscnn`, `CNN*` is the planar baseline. `Flat` and
`OnlyOneSpin` entries are ablations. `SCNN*` is `scnn`, registered only for
the two MNIST classification tasks, with widths cut to match `S3CNNMNIST`'s
parameter count; elsewhere build it directly with `arch="scnn"`.

Building a model directly instead:

```python
from modules.equivariant import SO3UNet

model = SO3UNet(arch="s3cnn", n_alpha=360, n_beta=179,
                resolutions=[179, 100, 50, 100, 179],
                widths=[1, 2, 8, 30, 8, 1],
                spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [1]])
```

> **Pick band-limits whose $2(L+1)$ factors into small primes.** $L = 100$ gives
> $202 = 2 \times 101$, and cuFFT is slow on prime sizes: it measured 73 ms of a
> 567 ms training step, about 60% of all transform time, for two layers out of
> five. $L = 99$ ($200 = 2^3\!\cdot\!5^2$) or $L = 101$ ($204 = 2^2\!\cdot\!3\cdot 17$)
> avoid it. The shipped schedules keep $100$ only because that is what the
> published results were trained at.
