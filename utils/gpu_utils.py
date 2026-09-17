"""Utils to pick and claim one idle GPU on a shared node.

Selection uses two signals. Utilisation alone is misleading: a busy job caught
between kernels reads 0%. Free memory alone is misleading too: a card can sit
at 0% while another user holds all of its memory. Both must pass.

Claiming runs in four steps: survey, lock, re-verify, pin. The survey takes
about a second, so a competitor can start inside that window, which is why
the re-verify exists.

The lock is advisory and only binds processes that import this module. Other
users on the node may ignore it.

Call this before torch initialises CUDA. Once a context exists the device set
is fixed.
"""

from __future__ import annotations

import atexit
import os
import time
from dataclasses import dataclass

_LOCK_DIR = os.environ.get("S3CNN_GPU_LOCK_DIR", "/tmp/s3cnn-gpu-locks")
_LOCK_STALE_S = 12 * 3600


@dataclass(frozen=True)
class GpuState:
    """One device: how busy it is, and how much of it belongs to someone else."""

    index: int
    uuid: str
    sm: int  # peak utilisation over the sampling window, percent
    other_mem: int  # bytes held by processes belonging to other users
    total_mem: int
    others: int  # number of such processes

    @property
    def mem_frac(self) -> float:
        return self.other_mem / self.total_mem

    def eligible(self, max_sm: int, max_mem_frac: float) -> bool:
        return self.sm < max_sm and self.mem_frac < max_mem_frac


def _visible_indices(count: int) -> list[int]:
    """Indices to survey, respecting a CUDA_VISIBLE_DEVICES set by SLURM.

    Only numeric tokens are parsed; if the variable holds UUIDs the whole node
    is surveyed instead.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return list(range(count))
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < count:
            out.append(int(tok))
    return out or list(range(count))


def survey(samples: int = 5, interval: float = 0.4) -> list[GpuState]:
    """Utilisation and foreign memory use of every visible device.

    Utilisation is the peak over `samples` readings, not the mean, so a job
    between kernels is not mistaken for an idle card. Memory is attributed by
    the uid owning each compute process: your own leftover processes do not
    count against a device, another user's do. A process that cannot be read
    counts as foreign.
    """
    import pynvml as nvml

    nvml.nvmlInit()
    try:
        handles = {i: nvml.nvmlDeviceGetHandleByIndex(i)
                   for i in _visible_indices(nvml.nvmlDeviceGetCount())}
        peak = {i: 0 for i in handles}
        for k in range(samples):
            if k:
                time.sleep(interval)
            for i, h in handles.items():
                peak[i] = max(peak[i], nvml.nvmlDeviceGetUtilizationRates(h).gpu)

        me, out = os.getuid(), []
        for i, h in handles.items():
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            foreign, n_foreign = 0, 0
            for p in nvml.nvmlDeviceGetComputeRunningProcesses(h):
                try:
                    if os.stat(f"/proc/{p.pid}").st_uid == me:
                        continue
                except OSError:
                    pass  # process gone or unreadable: count it, be conservative
                foreign += p.usedGpuMemory or 0
                n_foreign += 1
            out.append(GpuState(i, nvml.nvmlDeviceGetUUID(h), peak[i],
                                foreign, mem.total, n_foreign))
        return out
    finally:
        nvml.nvmlShutdown()


def _claim(uuid: str) -> bool:
    """Take an exclusive lock on a device. True if we got it.

    One file per GPU UUID, created with O_EXCL: of two processes racing for the
    same card, exactly one gets the descriptor and the other is told to look
    elsewhere. Keyed by UUID because the index is renumbered by
    CUDA_VISIBLE_DEVICES and is not stable between processes.

    A lock older than _LOCK_STALE_S is broken. A job killed with SIGKILL never
    runs its atexit hook, and would otherwise hold the device forever.
    """
    os.makedirs(_LOCK_DIR, exist_ok=True)
    path = os.path.join(_LOCK_DIR, f"{uuid}.lock")
    try:
        if time.time() - os.path.getmtime(path) > _LOCK_STALE_S:
            os.unlink(path)
    except OSError:
        pass
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    atexit.register(lambda: os.unlink(path) if os.path.exists(path) else None)
    return True


def format_survey(states: list[GpuState], max_sm: int, max_mem_frac: float) -> str:
    """One line per device, for logs and error messages."""
    rows = [f"{'gpu':>4} {'sm%':>5} {'others':>7} {'held':>9}  status"]
    for s in sorted(states, key=lambda s: s.index):
        ok = "free" if s.eligible(max_sm, max_mem_frac) else "busy"
        rows.append(f"{s.index:>4} {s.sm:>5} {s.others:>7} "
                    f"{s.other_mem / 2**30:>7.1f}G  {ok}")
    return "\n".join(rows)


def pick_gpu(max_sm: int = 10, max_mem_frac: float = 0.10, samples: int = 5,
             interval: float = 0.4, wait: float | None = None,
             force: int | None = None, verbose: bool = True) -> GpuState:
    """Claim one idle device and pin CUDA_VISIBLE_DEVICES to it.

    Candidates are ordered by foreign memory, then by utilisation, and the
    first one that can be locked is taken. A device someone else has locked is
    skipped, not an error.

    The claim is then checked once more against a fresh reading, since another
    user may have started during the survey. Only after that does the choice
    become final.

    CUDA_VISIBLE_DEVICES is set to the UUID rather than the index. The process
    then sees exactly one device, numbered cuda:0; `GpuState.index` keeps the
    node-level number and is only used for logging.

    Raises RuntimeError if nothing qualifies, after polling for `wait` seconds
    if one is given. `force` pins a device without checking or locking it, for
    reproduction runs.
    """
    deadline = time.time() + wait if wait else 0.0
    while True:
        states = survey(samples, interval)
        if force is not None:
            chosen = next((s for s in states if s.index == force), None)
            if chosen is None:
                raise RuntimeError(f"GPU {force} is not visible")
        else:
            free = [s for s in states if s.eligible(max_sm, max_mem_frac)]
            # least foreign memory first, then least busy
            free.sort(key=lambda s: (s.other_mem, s.sm))
            chosen = next((s for s in free if _claim(s.uuid)), None)

        if chosen is not None:
            # re-verify: a competitor may have started between survey and claim
            after = next(s for s in survey(1, 0) if s.uuid == chosen.uuid)
            if force is not None or after.eligible(max_sm, max_mem_frac):
                os.environ["CUDA_VISIBLE_DEVICES"] = chosen.uuid
                if verbose:
                    print(f"[gpu] claimed cuda:{chosen.index} {chosen.uuid} "
                          f"(sm {after.sm}%, {after.other_mem / 2**30:.1f}G held by others)")
                return chosen

        if time.time() >= deadline:
            raise RuntimeError(
                f"no GPU below {max_sm}% utilisation and {max_mem_frac:.0%} "
                f"foreign memory\n{format_survey(states, max_sm, max_mem_frac)}")
        time.sleep(min(30.0, max(5.0, deadline - time.time())))


def add_arguments(parser) -> None:
    """Standard --gpu / --max-sm / --max-mem-frac / --wait-gpu flags."""
    parser.add_argument("--gpu", type=int, default=None,
                        help="pin a specific device, skipping the idleness check")
    parser.add_argument("--max-sm", type=int, default=10,
                        help="reject devices busier than this percentage")
    parser.add_argument("--max-mem-frac", type=float, default=0.10,
                        help="reject devices with more than this fraction held by others")
    parser.add_argument("--wait-gpu", type=float, default=None,
                        help="poll for up to this many seconds instead of failing")


def from_args(args) -> GpuState:
    return pick_gpu(max_sm=args.max_sm, max_mem_frac=args.max_mem_frac,
                    wait=args.wait_gpu, force=args.gpu)


if __name__ == "__main__":
    states = survey()
    print(format_survey(states, 10, 0.10))
