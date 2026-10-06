# Overview

A classical CNN on a lat/lon grid is not equivariant to rotations of the
sphere as the grid distorts near the poles. The networks here are built from a
convolution defined on the group $SO(3)$ itself, so a rotation of the input
produces the same rotation of the output, up to quadrature error.

## The layer

Every equivariant network in this package is a stack of one block
(`modules/equivariant.Block`):

```
conv -> [normalisation] -> SO(3)-iFT -> nonlinearity -> dropout -> [smoothing] -> S2/SO(3)-FT
```

The convolution is a per-order contraction in the spectral domain. The nonlinearity is pointwise and
is performed int the spatial domain, which is why every block leaves the spectrum,
applies $\sigma$, and comes back. Those two transforms are the expensive part
of the model and [fused.md](fused.md) tries to tackle the computational cost by writing efficient kernels.

Between blocks the signal is a spectrum, and pooling is truncation of that
spectrum rather than a spatial operation.

## The three architectures

They differ in exactly one choice: *where* signal-equivariance is imposed.

| arch | equivariance | filter | parameters | $\sigma$ | after |
| ---- | ------------ | ------ | ---------- | -------- | ----- |
| `scnn` | at the output | $\psi^l_{m,n}$ | $O(L^3)$ | unconstrained | Cohen et al. 2018 |
| `s3cnn` | after every $\sigma$ | $\psi^l_n$ | $O(L^2)$ | unconstrained | **ours** |
| `swscnn` | inside the filter | $\psi^l$ | $O(L)$ | spin-preserving | Esteves et al. 2020, 2023 |

`scnn` and `s3cnn` both lift the signal to $SO(3)$. `s3cnn` projects back to $S^2$ every layer (the smoothing
operator) and `scnn` stays on the group until the output. `swscnn` never leaves
$S^2$, so its nonlinearity must preserve the spin weight by itself.

## Signals

Two kinds of field on the sphere are carried, distinguished by a spin weight
$n$: scalar fields ($n = 0$, e.g. temperature) and vector fields ($n = 1$,
e.g. wind, carried as one complex number per point). Both are stored as
$n$-equivariant signals on $SO(3)$, which is what lets one implementation serve
both.