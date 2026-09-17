"""End-to-end timings: forward, and forward plus backward, for the shipped
architectures at the resolutions their training scripts use.

Wigner tables are synthetic, as in `bench.ops`; loading the real ones costs
34 GB of pickle reads and does not change what the kernels do. The full table
is allocated only for the layer styles that read it, so peak memory means the
same thing in every row.

Batch norm is off everywhere: it is incompatible with the SO(3) layer styles
(defect 10), and leaving it on for the S2 styles only would make the columns
incomparable.
"""

from __future__ import annotations

import torch

from bench.harness import Result, time_call
from bench.ops import full_table, slice_tables
from modules.SO3CNN import SO3CNN
from modules.SO3UNet import SO3UNet

N_ALPHA, N_BETA = 360, 179

# Verbatim from run_training.py, plus a small one for smoke tests.
CONFIGS = {
    "prod": {
        "unet_so3": dict(cls=SO3UNet, layer_style="so3", fft_method="full",
                         resolutions=[179, 100, 50, 100, 179],
                         widths=[1, 2, 8, 30, 8, 1],
                         spins=[[1], [0, 1], [0, 1], [0, 1], [0, 1], [1]],
                         nonlinearity="leaky_relu"),
        "unet_spins": dict(cls=SO3UNet, layer_style="spins",
                           resolutions=[179, 150, 100, 50, 100, 150, 179],
                           widths=[1, 4, 32, 128, 256, 48, 16, 1],
                           spins=[[1]] + [[0, 1]] * 6 + [[1]],
                           nonlinearity="magnitude"),
        "cnn_so3": dict(cls=SO3CNN, layer_style="so3", fft_method="full",
                        resolution=179, widths=[1, 2, 4, 2, 1],
                        spins=[[1], [0, 1], [0, 1], [0, 1], [1]],
                        nonlinearity="leaky_relu"),
        "cnn_spins": dict(cls=SO3CNN, layer_style="spins", resolution=179,
                          widths=[1, 4, 32, 128, 256, 48, 16, 1],
                          spins=[[1]] + [[0, 1]] * 6 + [[1]],
                          nonlinearity="magnitude"),
    },
    "small": {
        "cnn_spins": dict(cls=SO3CNN, layer_style="spins", resolution=59,
                          widths=[1, 2, 2, 1], spins=[[1], [0, 1], [0, 1], [1]],
                          nonlinearity="magnitude"),
    },
}


def _build(spec, n_alpha, n_beta, L, device):
    """Only the SO(3) styles need the full table; allocating it anyway would
    put 11.1 GB into every peak-memory figure."""
    cls, kw = spec["cls"], {k: v for k, v in spec.items() if k != "cls"}
    n0, n1 = slice_tables(L, n_beta, device)
    full = full_table(L, n_beta, device) if kw["layer_style"] != "spins" else None
    return cls(n_alpha=n_alpha, n_beta=n_beta, use_batch_norm=False,
               wigner_n0=n0, wigner_n1=n1, f_wigner=full, **kw)


def cases(size: str, batch: int, iters: int, only: str | None = None):
    """Yield a forward and a forward+backward Result per configuration."""
    dev = torch.device("cuda")
    n_alpha = N_ALPHA if size == "prod" else 120
    n_beta = N_BETA if size == "prod" else 59
    L = max(max(s.get("resolutions", [s.get("resolution", 0)]))
            for s in CONFIGS[size].values())

    for name, spec in CONFIGS[size].items():
        if only is not None and only not in name:
            continue
        torch.manual_seed(0)
        model = _build(spec, n_alpha, n_beta, L, dev).to(dev)
        p = dict(B=batch, params=sum(q.numel() for q in model.parameters()))
        x = torch.randn(batch, n_alpha, n_beta, dtype=torch.complex64, device=dev)

        model.eval()
        with torch.no_grad():
            ms, std, peak = time_call(lambda: model(x), iters=iters)
        yield Result("models", f"{name}.forward", p, ms, std, peak, 0)

        model.train()

        def step():
            model.zero_grad(set_to_none=True)
            model(x).abs().sum().backward()

        ms, std, peak = time_call(step, iters=iters)
        yield Result("models", f"{name}.fwd+bwd", p, ms, std, peak, 0)

        del model, x
        torch.cuda.empty_cache()
