"""Benchmark entry point.

    python -m bench ops --size prod --out bench/baseline-ops.json
    python -m bench ops --baseline bench/baseline-ops.json

The GPU is claimed before torch is imported, so `CUDA_VISIBLE_DEVICES` is in
place by the time CUDA initialises.
"""

from __future__ import annotations

import argparse
import sys

from utils import gpu_utils


def _parse(argv):
    p = argparse.ArgumentParser("bench")
    p.add_argument("suite", choices=["ops", "models", "data"])
    p.add_argument("--size", default="prod", choices=["small", "prod"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--only", default=None,
                   help="run only cases whose name contains this string")
    p.add_argument("--out", default=None, help="write the run as JSON")
    p.add_argument("--baseline", default=None,
                   help="compare against a previous JSON and fail on a regression")
    p.add_argument("--tol", type=float, default=0.05,
                   help="fraction slower than the baseline that counts as a regression")
    gpu_utils.add_arguments(p)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    gpu = gpu_utils.from_args(args)

    import torch  # after the claim, so it sees one device

    from bench import data, harness, models, ops
    suite = {"ops": ops, "models": models, "data": data}[args.suite]

    def co_tenants():
        return next(s for s in gpu_utils.survey(1, 0) if s.uuid == gpu.uuid).others

    run = harness.Run(suite=args.suite, device=torch.cuda.get_device_name(0),
                      gpu_uuid=gpu.uuid, co_tenants=co_tenants(), co_tenants_end=0,
                      bandwidth_gb_s=harness.measure_bandwidth(torch.device("cuda")))
    run.results = list(suite.cases(args.size, args.batch, args.iters, args.only))
    run.co_tenants_end = co_tenants()

    baseline = harness.load(args.baseline) if args.baseline else None
    print(f"{run.device} {run.gpu_uuid} | {run.bandwidth_gb_s:.0f} GiB/s "
          f"| {run.co_tenants} other process(es)\n")
    print(harness.to_markdown(run, baseline))

    # The lock is advisory; another user can still land on the device mid-run.
    contended = run.co_tenants_end != run.co_tenants
    if contended:
        print(f"\nVOID: co-tenants went {run.co_tenants} -> {run.co_tenants_end} "
              f"during the run. Re-measure.")

    if args.out and not contended:
        run.to_json(args.out)
        print(f"\nwrote {args.out}")
    elif args.out:
        print(f"not writing {args.out}")

    if contended:
        return 2
    if baseline is not None:
        bad = harness.regressions(run, baseline, args.tol)
        if bad:
            print("\nregressions:\n  " + "\n  ".join(bad))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
