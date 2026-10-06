# Documentation

What the code assumes and why it is shaped the way it is. Full derivations and theory are in
the paper.

| file | contents |
| ---- | -------- |
| [overview.md](overview.md) | the problem, the layer, and the one choice the three architectures differ by |
| [notation.md](notation.md) | symbols and formulas: harmonics, Wigner $d$, the SO(3) transform, the smoothing operator |
| [conventions.md](conventions.md) | how signals, spectra and tables are laid out in tensors |
| [transforms.md](transforms.md) | the four transform modules, their normalisation, and the $\beta$ quadrature |
| [architectures.md](architectures.md) | `scnn`, `s3cnn`, `swscnn`, and the model zoo |
| [fused.md](fused.md) | the optimized fused operations and their kernels |
| [running.md](running.md) | install, precomputation, datasets, training, testing, benchmarks |