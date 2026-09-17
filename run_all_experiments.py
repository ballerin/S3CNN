"""Run a schedule of training and testing jobs, one at a time, on one GPU.

The schedule is whatever `train_schedule.create_schedule()` returns; see
`TEMPLATE_train_schedule.py` for the shape of an entry.

Jobs run strictly sequentially. The node is shared, so spreading a schedule
over several cards takes them from other users, and two jobs on one card make
each other's timings meaningless -- `bench/` numbers are only comparable
against a device running one thing.

One device is claimed before torch is imported and held for the whole
schedule. Each job then runs in its own subprocess: its CUDA context and its
copy of the dataset go back to the OS when it ends, and a job that dies takes
only itself down instead of the rest of the queue.

    python run_all_experiments.py --dry-run     # list the schedule, claim nothing
    python run_all_experiments.py               # claim an idle GPU and run it
    python run_all_experiments.py --gpu 3       # pin a device, skipping the check
    python run_all_experiments.py --schedule reproduce_schedule
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import sys
import time

from utils import gpu_utils
from utils import params as params_utils

OPERATIONS = ("train", "test")


def _run_job(params: dict) -> None:
    """Body of one scheduled job, in a fresh interpreter.

    `CUDA_VISIBLE_DEVICES` holds exactly one device, so the job's device is
    always cuda:0 whichever card the parent claimed.
    """
    # Imported here rather than at module scope: the parent must not touch CUDA.
    from run_testing import run_testing
    from run_training import run_training
    from utils.precision import enable_tf32

    enable_tf32()

    {"train": run_training, "test": run_testing}[params["operation"]](
        gpu_id=0, params=params)


def describe(params: dict) -> str:
    """A job in one line, for the queue listing and the summary."""
    keys = ("operation", "experiment", "model_type", "nonlinearity", "rotate",
            "dataset_size", "log_id")
    return " ".join(str(params[k]) for k in keys if params.get(k) is not None)


def validate(tasks: list[dict]) -> None:
    """Reject a malformed schedule before a GPU is claimed for it.

    Keys are checked here, for every job, rather than when each job starts:
    a typo in job 47 should not be found six hours into the queue. Importing
    the entry points pulls in torch, which is harmless in this process -- it
    surveys devices through NVML and never creates a CUDA context.
    """
    from run_testing import run_testing
    from run_training import run_training
    consumer = {"train": run_training, "test": run_testing}

    for i, params in enumerate(tasks, 1):
        op = params.get("operation")
        if op not in OPERATIONS:
            raise ValueError(f"job {i}: 'operation' must be one of {OPERATIONS}, "
                             f"got {op!r}")
        try:
            params_utils.check_keys(params, consumer[op])
        except ValueError as exc:
            raise ValueError(f"job {i}: {exc}") from None


def run_schedule(tasks: list[dict], target=_run_job) -> list[tuple[str, int, float]]:
    """Run every job in turn. Returns (label, exit code, seconds) per job.

    A failing job is reported and the queue carries on: an overnight schedule
    should not be lost to one bad set of parameters. `target` is the per-job
    entry point; the tests substitute a cheap one for it.
    """
    ctx = mp.get_context("spawn")  # a forked CUDA context is not usable
    out = []
    for i, params in enumerate(tasks, 1):
        label = describe(params)
        print(f"\n=== [{i}/{len(tasks)}] {label} ===", flush=True)
        started = time.monotonic()
        proc = ctx.Process(target=target, args=(params,), name=f"job{i}")
        proc.start()
        try:
            proc.join()
        except KeyboardInterrupt:
            proc.terminate()
            proc.join()
            print(f"\ninterrupted during job {i}; {len(tasks) - i} not run")
            raise
        elapsed = time.monotonic() - started
        status = "ok" if proc.exitcode == 0 else f"FAILED rc={proc.exitcode}"
        print(f"=== [{i}/{len(tasks)}] {status} in {elapsed / 60:.1f} min ===",
              flush=True)
        out.append((label, proc.exitcode, elapsed))
    return out


def summarise(results: list[tuple[str, int, float]]) -> str:
    width = max((len(label) for label, _, _ in results), default=0)
    rows = [f"{label.ljust(width)}  {elapsed / 60:>7.1f} min  "
            f"{'ok' if code == 0 else f'FAILED rc={code}'}"
            for label, code, elapsed in results]
    total = sum(elapsed for _, _, elapsed in results)
    failed = sum(1 for _, code, _ in results if code != 0)
    rows.append(f"{len(results)} job(s), {failed} failed, "
                f"{total / 3600:.1f} h total")
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        "run_all_experiments", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="list the schedule and exit without claiming a GPU")
    parser.add_argument("--schedule", default="train_schedule",
                        help="module providing create_schedule() (default: train_schedule)")
    gpu_utils.add_arguments(parser)
    args = parser.parse_args(argv)

    create_schedule = importlib.import_module(args.schedule).create_schedule

    tasks = list(create_schedule())
    validate(tasks)
    if not tasks:
        print("schedule is empty")
        return 0

    print(f"{len(tasks)} job(s) queued:")
    for i, params in enumerate(tasks, 1):
        print(f"  {i:>3}. {describe(params)}")
    if args.dry_run:
        return 0

    gpu_utils.from_args(args)  # held until this process exits

    results = run_schedule(tasks)
    print("\n" + summarise(results))
    return 1 if any(code != 0 for _, code, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
