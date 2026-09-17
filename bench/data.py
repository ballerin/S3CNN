"""Input-pipeline throughput against `num_workers`.

Defect 16 asks what `num_workers` should be. The repo has two regimes and
they want opposite answers, which is why one number was never right:

*cached* -- `preload_dataset=True`, the default, and unconditional for the
three ERA5 experiments. The whole dataset is a list of CPU tensors in RAM and
`__getitem__` is a list index. A worker cannot make an index faster, but it
does have to pickle every sample back through a pipe, and it is forked afresh
at every epoch. Memory is not the objection -- workers are forked, so the
tensor storage is shared copy-on-write and the measured cost of eight of them
is ~130 MiB, not eight copies -- but time is: they are both slower per batch
and hundreds of milliseconds more expensive per epoch.

*uncached* -- `preload_dataset=False`. `__getitem__` projects an MNIST digit
onto the sphere (`stitch_mnist_to_sphere_*`, plus `compute_gradients` for the
vector input), which is real NumPy work per sample and is exactly what
workers are for.

Two numbers per configuration. `startup` is the first batch out of a fresh
iterator, which is what an epoch boundary costs: workers are forked, and each
one imports and inherits before it answers. `steady` is the mean of the
batches after that. Training pays the first once per epoch and the second
once per step.
"""

from __future__ import annotations

import gc

import torch

from bench.harness import Result, rss_tree, time_wall
from SphericalDataset import SphericalMNISTDataset, WeatherDataset

WORKERS = (0, 2, 4, 8)

# The val splits: same __getitem__, same tensor shapes, a tenth of the wait.
# `batches` must clear `max(WORKERS) * prefetch_factor` by a wide margin, or
# the queue is filled during startup and the steady figure measures a dequeue.
SIZES = {"small": dict(batch=8, batches=64), "prod": dict(batch=32, batches=256)}


def _mnist(caching: bool):
    return SphericalMNISTDataset(
        n_alpha=120, n_beta=59, split="val", mode="spherical",
        in_datatype="scalar", out_datatype="label", rotation_type="no",
        caching=caching, verbose=False)


def _weather():
    """Ten years of ERA5. One year is 52 weekly samples, which at any batch
    size the schedule uses is an epoch of one or two steps."""
    ds = WeatherDataset(root="dataset/wind->wind/weekly_coarse_torch/",
                        folders=[str(y) for y in range(2000, 2010)], train=False,
                        in_datatype="vector", out_datatype="vector")
    ds.preload()
    return ds


def _drain(dataset, batch_size: int, workers: int, batches: int, dev):
    """(startup ms, steady ms, steady std, resident bytes) for one loader.

    A fresh iterator per sample, because the cost being measured is the one an
    epoch pays; `persistent_workers` would hide it and training does not use
    it. The batch is moved to the device, since `pin_memory` is only worth
    anything against a real host-to-device copy. Memory is read mid-epoch,
    while the workers are alive and holding whatever they hold.
    """
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=workers)

    def first():
        it = iter(loader)
        x, _ = next(it)
        x.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        del it  # shut the workers down before the next sample is timed

    startup, _ = time_wall(first, warmup=1, iters=3)

    resident = 0

    def rest():
        nonlocal resident
        it = iter(loader)
        next(it)
        for i in range(batches):
            x, _ = next(it)
            x.to(dev, non_blocking=True)
            if i == batches // 2:
                resident = max(resident, rss_tree())
        torch.cuda.synchronize()
        del it

    total, std = time_wall(rest, warmup=1, iters=3)
    return startup, (total - startup) / batches, std / batches, resident


def cases(size: str, batch: int, iters: int, only: str | None = None):
    """Yield a startup and a steady Result per dataset and worker count."""
    cfg = SIZES[size]
    batch_size = batch if batch > 1 else cfg["batch"]
    dev = torch.device("cuda")

    sources = {"mnist_cached": lambda: _mnist(True),
               "mnist_uncached": lambda: _mnist(False),
               "weather_cached": _weather}

    for name, build in sources.items():
        if only is not None and only not in name:
            continue
        dataset = build()
        # One short of a full pass: `next` past the end would raise.
        batches = max(1, min(cfg["batches"], len(dataset) // batch_size - 1))
        if batches < 2 * max(WORKERS):
            print(f"  {name}: {len(dataset)} samples is too few at B={batch_size}; "
                  f"steady times would measure the prefetch queue")
        for workers in WORKERS:
            gc.collect()
            startup, steady, std, resident = _drain(
                dataset, batch_size, workers, batches, dev)
            p = dict(workers=workers, B=batch_size, n=batches)
            yield Result("data", f"{name}.startup", p, startup, 0.0, resident, 0)
            yield Result("data", f"{name}.steady", p, steady, std, resident, 0)
        del dataset
        gc.collect()
