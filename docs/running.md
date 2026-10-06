# Running the code

## Install

```
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

Versions are pinned in `pyproject.toml` — they are what the measurements were
taken with. `requirements.txt` is the same list without pins. Python ≥ 3.12.

## Precomputation

The Wigner tables are computed once and read from `precomp/` (git-ignored):

```
python generate_precomps.py --n_beta 179 --l_max 179 --compute_wigner
python generate_precomps.py --rotations --n_rotations 1000
```

| file | contents |
| ---- | -------- |
| `SLICE_WIGNER_N0_<n_beta>_<l_max>.list` | $d^l_{m,n}$, column $n = 0$ |
| `SLICE_WIGNER_N1_<n_beta>_<l_max>.list` | $d^l_{m,n}$, column $n = 1$ |
| `FULL_WIGNER_<n_beta>_<l_max>.list` | the full $d^l_{m,n}$ blocks |
| `BDELTAS_<n_beta>_<l_max>.list` | $\Delta^l = d^l(\pi/2)$, for the $\Delta$ method |
| `CONJ_PROD_<n_beta>_<l_max>.list` | products $\Delta^l_{s,m}\Delta^l_{s,n}$ |
| `SINES_<n_beta>.torch` | $\sin\beta$ on the grid |
| `ROTATIONS_<n>.pckl` | random rotations used at test time |

`l_max` usually equals `n_beta` (Nyquist), but may be lower when a higher
latitude resolution is wanted for spectral rotations. Layout and packing:
[conventions.md](conventions.md#wigner-tables). The directory is large — tens
of gigabytes at production resolution — and loading it dominates start-up.

Only the `FULL_WIGNER` and `SLICE_WIGNER` tables are read by the models;
`BDELTAS`, `CONJ_PROD` and `SINES` belong to the unimplemented `"delta"`
transform ([transforms.md](transforms.md#normalisation)).

## Data

### ERA5

Downloaded separately and placed under `dataset/`, organised by experiment type
and then by year, with `X` and `Y` directories of `.npy` files per year:

```
dataset/<type>/<year>/{X,Y}/<index>.npy
```

`<type>` may be nested, and encodes both the task and the coarseness
(`wind->wind/weekly_coarse_torch/`). Read by `SphericalDataset.WeatherDataset`;
the roots are in `modules/zoo.py`.

### Spherical MNIST

Not stored locally — fetched through torchvision on first use by
`SphericalMNISTDataset(split=..., mode="equirectangular", n_alpha=..., n_beta=...)`,
then stitched onto the sphere by `SO3lib.stitch_mnist_to_sphere_*`.

### Experiments

| key | task |
| --- | ---- |
| `wind->wind` | ERA5 wind (vector) to T+24h wind (vector) |
| `temp->wind` | ERA5 temperature (scalar) to T+0h wind (vector) |
| `autoencoder` | ERA5 wind compressed to at most 512 float32 |
| `MNIST-scalar-classification` | Spherical MNIST |
| `MNIST-vf-classification` | Spherical MNIST with a Sobel filter |
| `MNIST-scalar-to-vector` | Spherical MNIST to Spherical Sobel MNIST |
| `MNIST-vector-to-scalar` | the reverse |

## Training and testing

Write the jobs into `train_schedule.py` (copy `TEMPLATE_train_schedule.py`,
which is git-ignored so local schedules stay local) and run:

```
python run_all_experiments.py --GPU 0
python run_all_experiments.py --GPU 2,3,5     # one job per GPU, in parallel
```

A schedule entry is a dict; `operation` is `"train"` or `"test"`, and the rest
are the parameters of `run_training.train` / `run_testing`:

| key | values | default |
| --- | ------ | ------- |
| `model_type` | a key of `modules/zoo.py` | — |
| `experiment` | one of the table above | — |
| `total_epochs`, `early_stopping`, `save_every`, `batch_size` | positive ints | — |
| `lr` | Adam learning rate | `1e-3` |
| `nonlinearity` | overrides the architecture's own | architecture default |
| `use_batch_norm`, `use_layer_norm` | bool | `False` |
| `recompute` | rebuild each slab in the backward pass ([fused.md](fused.md#recompute)) | `True` |
| `fft_method` | `"full"` (only one implemented) | `"full"` |
| `dataset_size` | training years, 1–10 (weather only) | — |
| `rotate` | `"no"`, `"offset"`, `"beta"`, `"full"` | — |
| `gradient_accumulation` | optimiser step every this many batches | `1` |
| `pretrained_weights`, `starting_epoch` | resume from a checkpoint | — |
| `log_id` | string identifying the run | — |

> `lr` is **not tuned** for the current code. `1e-3` was chosen when the
> convolution kernels were initialised as a real constant; complex Glorot
> initialisation replaced that and raises the output RMS at initialisation
> 40–200x. Sweep it before comparing against published numbers.

Logs, loss curves and the best checkpoint land under
`experiments/<experiment>/<mode>/<rotation>/[<dataset_size>/]<log_id>/`, where
`<mode>` names the architecture and `<rotation>` is `NR` (`no`, `offset`), `RB`
(`beta`) or `RF` (`full`). Initialise the empty directories if a run complains.
`--schedule` names a different schedule module; `--dry-run` lists the jobs
without claiming a GPU.

A `"test"` job needs its training job to have run first, and takes
`model_type`, `experiment`, `batch_size`, `rotate`, `fft_method` and `path` —
the relative path of the training folder above.

To run one pair end to end — precompute, train, test — on a single claimed GPU:

```
./reproduce.sh wind->wind S3CNNUNet 1 paper
```

Plotting scripts that read run ids are in `plotting/`.

## Tests and benchmarks

```
pytest                     # correctness, small sizes, CPU
python -m bench ops        # per-operator timings
python -m bench models     # forward and forward+backward per model
```

The suite builds its Wigner tables from `tests/reference.py` rather than
`precomp/`, so it runs on a fresh clone, and pins outputs against
`tests/goldens.npz`. Strict xfails mark known defects — notably the quadrature
at $L = n_\beta$ ([transforms.md](transforms.md#quadrature-over-beta)); they
are tripwires, and turning one green is a deliberate act.

`bench` claims an idle GPU through `utils/gpu_utils.py`, counts foreign
processes at both ends of a run and refuses to write its JSON if any appeared.
The claim is advisory: it stops your own runs colliding, not other users'.
`--out` writes a run, `--baseline` compares against one and fails on a
regression beyond `--tol`.
