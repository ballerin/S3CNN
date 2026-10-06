# The fused $SO(3)$ path

`modules/fused.py`, `modules/kernels.py`. This is an optimisation: it changes
where the work happens, not what is computed. `tests/test_fused.py` checks it
against the plain path, which `fuse=False` restores.

## The problem

Unfused, a lifting block walks through three tensors the size of the $SO(3)$
grid — `[batch, channels, n_alpha, n_beta, n_gamma]`, gigabytes at production
resolution — and keeps the last alive for the backward pass, all so that a
pointwise nonlinearity can be applied and the $\gamma$ axis integrated away
immediately afterwards.

## Slabs

A **slab** is a contiguous block of latitudes: a slice of the $\beta$ axis, with
the full $\alpha$ and $\gamma$ axes at those latitudes.

Nothing after the $\beta$-synthesis couples different latitudes — the transforms
over $\alpha$ and $\gamma$, the nonlinearity, the dropout and the integral over
$\gamma$ all act within one $\beta$. So the whole chain runs on one slab at a
time and the grid never exists at full size:

```
synthesis -> sigma -> dropout -> smoothing    (fused.project,  s3cnn)
synthesis -> sigma -> dropout -> analysis     (fused.regroup,  scnn)
```

$\beta$ is the axis to cut, not $\gamma$: evaluating a single $\gamma$ needs
every coefficient in $n$, so a $\gamma$ slab would still require the whole grid.

`fused.slab()` converts a byte budget (`SLAB_BYTES`, 256 MiB) into a latitude
count, so a slab gets thinner as the batch grows. The thickness changes nothing
about the result.

## The kernels

The $\beta$-synthesis

$$g[m, \beta, n] = \sum_{l \ge \max(|m|,|n|)} \hat x^l_{mn}\; d^l_{mn}(\beta)$$

written with tensor operations is one pass per order, each adding into a
sub-block, so a cell near the middle is read and written about $L$ times —
tens of gigabytes of traffic for a grid of one. Padding the orders into a
rectangle lets a single matmul do it, at the price of a padded Wigner table
roughly three times the size.

A Triton kernel needs neither: the orders live in one packed buffer at offset
$l(4l^2-1)/3$, so a program walks them in place, keeps its running sum in
registers, and writes each output cell exactly once. The tables are real, so a
complex multiply is two accumulations rather than four. The kernel writes at
the `fftshift`ed position directly — done afterwards it is a `roll`, another
full copy of the grid.

The gradient of the synthesis is a sum over latitudes at fixed order, which is
the analysis contraction, so the two kernels in `modules/kernels.py` are each
other's backward. Without Triton, or off the GPU, `_accumulate` is the
reference path and gives the same answer.

## Recompute

By default each slab is rebuilt in the backward pass instead of kept
(`torch.utils.checkpoint`, `use_reentrant=False`). What that keeps alive is not
the $SO(3)$ grid but the checkpoint's inputs — one packed buffer per block,
shared by every slab of it — so retention does not scale with the slab count,
and peak memory is set by that buffer rather than by `SLAB_BYTES`.

`recompute=False` keeps the slabs instead: same answer, less time, more memory.
Gradients are bit-identical either way, in train mode with dropout live as well
as in eval, because the RNG state is restored and the recompute redraws the
same masks. So the flag is safe to set per run according to what the card has
free — see the table beside `fused.SLAB_BYTES` for the measured cost, about
1.3x, and the batch size at which keeping them stops fitting.

Two cases skip the checkpoint whatever the flag says: inference, where there is
no graph to keep alive, and a block handed inputs that do not require grad —
which is the first block of a network, on the reference path where the
convolution is left unevaluated.
