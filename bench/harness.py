"""Timing, memory and reporting for the benchmark suites.

Every measurement is taken on one GPU claimed by `utils.gpu_utils` before
torch initialises CUDA. That claim is advisory, so the number of foreign
processes is recorded at the start and again at the end: if it changed,
someone else was on the device and the run has to be thrown away.

Times come from CUDA events around a synchronised loop; memory from
`torch.cuda.max_memory_allocated`, which counts the caching allocator's peak
rather than the reserved pool. The roofline column is the time the same
traffic would take at the bandwidth this device actually reaches on a large
copy, so it is a floor, not a target.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass, field

import torch

BYTES_COMPLEX64 = 8
BYTES_FLOAT32 = 4


@dataclass
class Result:
    suite: str
    name: str
    params: dict
    ms: float
    ms_std: float
    peak_mem: int
    traffic: int  # bytes that must move regardless of implementation

    def roofline_ms(self, gb_per_s: float) -> float:
        return 1e3 * self.traffic / (gb_per_s * 2**30)


@dataclass
class Run:
    """One invocation of a suite: the results plus what they were measured on."""

    suite: str
    device: str
    gpu_uuid: str
    co_tenants: int  # foreign processes on the device when the run started
    co_tenants_end: int  # ... and when it finished; a change invalidates the run
    bandwidth_gb_s: float
    torch: str = torch.__version__
    host: str = field(default_factory=platform.node)
    results: list = field(default_factory=list)

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=1)


def measure_bandwidth(device, nbytes: int = 1 << 28) -> float:
    """Achieved device bandwidth in GiB/s, from a large device-to-device copy."""
    a = torch.empty(nbytes // 4, dtype=torch.float32, device=device)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(20):
        b.copy_(a)
    stop.record()
    torch.cuda.synchronize()
    return 20 * 2 * nbytes / (start.elapsed_time(stop) * 1e-3) / 2**30


def time_call(fn, warmup: int = 3, iters: int = 10) -> tuple[float, float, int]:
    """Mean and standard deviation in ms, and the allocator's peak in bytes."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(iters):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))

    t = torch.tensor(samples)
    return t.mean().item(), t.std().item(), torch.cuda.max_memory_allocated()


def time_wall(fn, warmup: int = 1, iters: int = 10) -> tuple[float, float]:
    """Mean and standard deviation of `fn` in ms, on the host clock.

    For work that is not a stream of CUDA kernels -- the input pipeline, where
    the cost is `__getitem__`, pickling between processes and the host copy --
    and where a CUDA event would only time the tail of it.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        samples.append(1e3 * (time.perf_counter() - started))
    t = torch.tensor(samples)
    return t.mean().item(), t.std().item() if len(samples) > 1 else 0.0


def _proportional_set(pid) -> int:
    """One process's share of resident memory, in bytes.

    Pss, not Rss: a page shared by N processes counts 1/N times in each. That
    is the distinction the DataLoader question turns on, because its workers
    are forked and inherit a preloaded dataset copy-on-write -- Rss would bill
    every worker for the whole of a dataset none of them actually copied.
    Falls back to Rss where smaps_rollup is not readable.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    with open(f"/proc/{pid}/statm") as fh:
        return int(fh.read().split()[1]) * resource.getpagesize()


def rss_tree() -> int:
    """This process and its live children, in bytes, read right now.

    An instantaneous reading rather than `ru_maxrss`, which is a high-water
    mark: it saturates after the first case and reports nothing about the
    ones that follow. It must therefore be taken while the children are
    alive.
    """
    pid = os.getpid()
    total = _proportional_set(pid)
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as fh:
            children = fh.read().split()
    except OSError:
        return total
    for child in children:
        try:
            total += _proportional_set(child)
        except OSError:
            pass  # exited between listing and reading
    return total


def _fmt_mem(n: int) -> str:
    return f"{n / 2**30:.2f}G" if n >= 2**30 else f"{n / 2**20:.0f}M"


def to_markdown(run: Run, baseline: Run | None = None) -> str:
    # The roofline only means something where a traffic estimate was given.
    roofline = any(r.traffic for r in run.results)
    head = ["case", "ms", "±", "peak"]
    if roofline:
        head += ["roofline", "of peak"]
    if baseline is not None:
        head.append("vs base")
    prev = {_key(r["name"], r["params"]): (r["ms"], r["params"].get("params"))
            for r in (baseline.results if baseline else [])}

    rows, resized = [], []
    for r in run.results:
        row = [f"{r.name} {_params(r.params)}", f"{r.ms:.2f}", f"{r.ms_std:.2f}",
               _fmt_mem(r.peak_mem)]
        if roofline:
            floor = r.roofline_ms(run.bandwidth_gb_s)
            row += [f"{floor:.2f}", f"{floor / r.ms:.1%}"]
        if baseline is not None:
            was = prev.get(_key(r.name, r.params))
            row.append("--" if was is None else f"{r.ms / was[0] - 1:+.1%}")
            if was is not None and was[1] not in (None, r.params.get("params")):
                resized.append(f"{r.name}: {was[1]} -> {r.params['params']} parameters")
        rows.append(row)

    width = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(head)]
    out = [" | ".join(h.ljust(w) for h, w in zip(head, width)),
           "-|-".join("-" * w for w in width)]
    out += [" | ".join(c.ljust(w) for c, w in zip(row, width)) for row in rows]
    if resized:
        out += ["", "The model itself changed since the baseline:"]
        out += ["  " + line for line in dict.fromkeys(resized)]
    return "\n".join(out)


def _params(p: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in p.items())


def _key(name: str, params: dict):
    """What makes two measurements comparable: the case and the knobs it was
    swept over. `params` is the model's parameter count -- a property of the
    model, not a knob -- and it moves whenever the architecture is fixed, which
    would otherwise silently drop every row from the comparison."""
    return name, json.dumps({k: v for k, v in params.items() if k != "params"},
                            sort_keys=True)


def regressions(run: Run, baseline: Run, tol: float) -> list[str]:
    prev = {_key(r["name"], r["params"]): r["ms"] for r in baseline.results}
    out = []
    for r in run.results:
        was = prev.get(_key(r.name, r.params))
        if was and r.ms > was * (1 + tol):
            out.append(f"{r.name} {_params(r.params)}: {was:.2f} -> {r.ms:.2f} ms")
    return out


def load(path) -> Run:
    with open(path) as fh:
        return Run(**json.load(fh))
