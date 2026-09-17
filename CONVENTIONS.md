# Conventions

How signals, spectra and Wigner tables are encoded across the package. The
derivations behind them are in the paper; this file states only what the code
assumes.

## Spins

`n` distinguishes the two kinds of n-equivariant signal on SO(3):

| n | signal | example |
| - | ------ | ------- |
| 0 | scalar field on S2 | temperature |
| 1 | vector field on S2 | wind |

They are n-equivariant in the sense that `f(A Z(gamma)) = f(A) e^{-i n gamma}`
for `A` in SO(3), which is what lets an S2 signal be carried as a slice of an
SO(3) one.

## Spectra on S2

A band-limited spectrum is one flat tensor `[batch, channels, (L+1)^2]`. Order
`l` occupies the slice `l**2 : (l+1)**2`, which is `2l+1` long: consecutive
squares differ by consecutive odd numbers.

Spin 1 has no `l = 0` term, so its natural length is one shorter, with order
`l` at `(l-1)*(l+1) : l*(l+2)`. Both spins are nevertheless stored at the
spin-0 length, with **a dummy zero as the first coefficient** when `n = 1`.
Tensors then stay homogeneous when a layer carries both spins — half the
channels each — at the cost of one wasted coefficient.

## Spectra on SO(3)

A general SO(3) spectrum is a **list**, one entry per order, where entry `l`
has shape `[batch, channels, 2l+1, 2l+1]` indexed `(m, n)`. Truncating the list
band-limits the signal, which is how the group path pools
(`modules/equivariant.resize`).

`SO3_to_S2` takes column `n` of every block and returns the flat S2 layout
above; `S2_to_SO3` pads the other way.

## Grids

Spatial signals are sampled on `n_alpha` longitudes and `n_beta` latitudes,
excluding the poles, with `beta_j = pi (j+1) / (n_beta+1)`
(`SO3lib.from_index_to_beta`). A layer band-limited at `L` uses
`n_alpha = n_gamma = 2(L+1)`. An SO(3) spatial signal carries a third axis:
`[batch, channels, alpha, beta, gamma]`.

The quadrature over beta is a sine-weighted rectangle rule with weight
`pi/(n_beta+1) * sin(beta)`. It is **not** exact at `L = n_beta`, the shipped
setting; see the quadrature study (defect 2, milestone M6). The rule lives in
`modules/quadrature.py` and the analysis transforms take a `quadrature=`
argument, so a replacement is registered there rather than edited into
`SO3fft.py`.

## Transform normalisation

Analysis and synthesis are inverse up to quadrature error, with alpha and gamma
following `torch.fft` conventions: `norm="ortho"` over alpha in both
directions; over gamma, `SO3iFFT` leaves the transform unnormalised and
`SO3FFT` uses `norm="forward"`, so that `SpatialSmoothing`'s average over gamma
returns the coefficient itself.

## Wigner tables

`d^l_{mn}(beta)` is real, and is stored as such -- real fp32, not complex64,
which halves both the bytes and the multiply count (`SO3lib.contract` threads
the (re, im) axis through the contraction). One table per order:

| table | file | contents |
| ----- | ---- | -------- |
| full | `FULL_WIGNER_<n_beta>_<l_max>.list` | `[n_beta, 2l+1, 2l+1]` per `l` |
| spin 0 | `SLICE_WIGNER_N0_<n_beta>_<l_max>.list` | column `n = 0` |
| spin 1 | `SLICE_WIGNER_N1_<n_beta>_<l_max>.list` | column `n = 1`, from `l = 1` |

The slices are what the S2 transforms need; only the group path reads the full
table. Each table is packed into **one** buffer (`precomp.Packed`), with
`tables[l]` a view into it: the orders used to be `L+1` separate buffers
rebuilt into a Python list on every call, and the ragged full table used
`torch.nested`, a prototype API that blocks `torch.compile`. Buffers move with
`.to(device)` but stay out of the `state_dict` (`persistent=False`).
`modules/precomp.py` also accepts the older `<kind>_<n_beta>.list` names, and
converts a complex table on load.

## References

- Kostelec & Rockmore 2008, *FFTs on the rotation group* — the SO(3) transform
  and the separation of variables.
- Cohen et al. 2018, *Spherical CNNs* — the sCNN preset.
- Esteves et al. 2020, 2023 — spin-weighted spherical CNNs, the swsCNN preset.
- Trabelsi et al. 2018, *Deep Complex Networks*, sec. 3.6 — the complex Glorot
  initialisation in `SO3ConvLayer.complex_glorot`.
