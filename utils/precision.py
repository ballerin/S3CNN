"""Matmul precision on the GPU.

TF32 keeps fp32 range with a 10-bit mantissa and runs on tensor cores, which is
substantially faster for the matmuls inside the transforms on hardware that has
them. It is set here rather than scattered through the code, so a run can turn
it off in one place and a precision study has one thing to toggle.

Equivariance is the metric that decides whether reduced precision is
acceptable; see `tests/test_models.py::test_z_rotation_equivariance`.
"""

from __future__ import annotations

import os

import torch

#: Set S3CNN_TF32=0 to keep full fp32 matmuls.
ENV = "S3CNN_TF32"


def enable_tf32(enabled: bool | None = None) -> bool:
    """Turn TF32 on (the default) or off. Returns what was set."""
    if enabled is None:
        enabled = os.environ.get(ENV, "1") != "0"
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    return enabled
