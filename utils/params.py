"""Checking a schedule entry against the function that will consume it.

`run_training` and `run_testing` read their parameters with `params.get(...)`,
so a misspelled key is not an error -- it is a default, silently substituted.
`TEMPLATE_train_schedule.py` carried `"pretrained_wights"` for exactly that
reason: the job ran, from scratch, and said nothing.

The accepted names are read off the consuming function's own signature, so
there is no second list here to drift out of date.
"""

from __future__ import annotations

import difflib
import inspect

#: Keys the runner reads before a job is handed to either entry point.
RUNNER_KEYS = {"operation"}


def accepted(fn) -> set[str]:
    """Parameter names `fn` takes, minus the two the runner supplies."""
    return set(inspect.signature(fn).parameters) - {"gpu_id", "params"}


def check_keys(params: dict, fn) -> None:
    """Raise if `params` holds a key `fn` would ignore."""
    known = accepted(fn) | RUNNER_KEYS
    unknown = sorted(set(params) - known)
    if not unknown:
        return
    lines = []
    for key in unknown:
        near = difflib.get_close_matches(key, known, n=1, cutoff=0.7)
        lines.append(f"  {key!r}" + (f" -- did you mean {near[0]!r}?" if near else ""))
    raise ValueError(
        f"{fn.__name__} does not take:\n" + "\n".join(lines) +
        f"\nAccepted: {', '.join(sorted(known))}")
