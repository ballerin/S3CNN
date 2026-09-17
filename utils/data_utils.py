"""One place where a DataLoader is configured.

`num_workers` was left at its default everywhere, which is right for half the
call sites and wrong for the other half, because the two halves ask
`__getitem__` for completely different things:

- A **preloaded** dataset -- `preload_dataset=True`, the default, and
  unconditional for the three ERA5 experiments -- is a list of CPU tensors and
  `__getitem__` is an index into it. A worker cannot index faster than the
  main process, it has to pickle every sample back through a pipe, and it is
  forked afresh each epoch. Zero workers is right.
- Anything else does real per-sample work. Spherical MNIST built on the fly
  projects a digit onto the sphere per item; that is NumPy on the host and it
  parallelises.

`python -m bench data` is the measurement; `bench/data.py` says what it
measures and why.
"""

from __future__ import annotations

import os

import torch

#: Enough, and not a tuning knob. A model step here is 1.5-3.2 s; four workers
#: already put the heaviest `__getitem__` in the repo at ~1% of that, and the
#: eight-worker configuration buys back 18 ms of it for twice the forks.
#: Measured by `python -m bench data`.
MAX_WORKERS = 4


def available_cpus() -> int:
    """Cores this process may actually run on, not cores the node has.

    The affinity mask is what a batch system hands out; on a shared node
    `os.cpu_count()` is an invitation to oversubscribe someone else's job.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def worker_count(dataset, requested: int | None = None) -> int:
    """Workers to give a loader over `dataset`.

    `requested` overrides the policy but is still capped by the affinity mask.
    """
    if requested is None:
        requested = 0 if getattr(dataset, "is_preloaded", False) else MAX_WORKERS
    return max(0, min(requested, available_cpus() - 1))


def make_dataloader(dataset, batch_size: int, shuffle: bool,
                    workers: int | None = None, **kwargs):
    """A DataLoader with the worker policy applied.

    `pin_memory` throughout: every batch goes straight to the GPU, and a
    pinned staging buffer is what makes that copy asynchronous.

    Workers, where there are any, are persistent. An epoch here is short --
    ten years of weekly ERA5 is 520 samples, sixteen steps at batch 32 -- and
    a fork costs more than the epoch it was meant to speed up.
    """
    n = worker_count(dataset, workers)
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=n, persistent_workers=n > 0, **kwargs)
