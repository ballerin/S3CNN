# Conventions

How signals, spectra and Wigner tables are encoded across the package.

## Spins

$n$ is at the same time both the index within the order of the Fourier coefficients for a spherical signal, and the "type" of spherical signal whenever all other coefficients are zero. In that case $n$ distinguishes the two kinds of $n$-equivariant signal on $SO(3)$ that we use:

| $n$ | signal | example |
| - | ------ | ------- |
| 0 | scalar field on $S^2$ | temperature |
| 1 | vector field on $S^2$ | wind |

They are $n$-equivariant in the sense that $f(A\,Z(\gamma)) = f(A)\,e^{-in\gamma}$
for $A \in SO(3)$, which is what lets an $S^2$ signal be carried as a slice of an
$SO(3)$ one.

## Spectra on $S^2$

A spectrum band-limited at $L$ has $(L+1)^2$ coefficients $c_{lm}$, with
$0 \le l \le L$ and $|m| \le l$. They can be laid out in two ways.

**Flat.** One axis of length `(L+1)**2`. Order $l$ occupies
`l**2 : (l+1)**2`, which is $2l+1$ long (consecutive squares differ by
consecutive odd numbers), and $c_{lm}$ sits at `l**2 + l + m`. There is no
padding.

**Padded.** A 2-D block with one axis for $l$ and one for $m$, zero wherever
$|m| > l$. The Esteves et al. spin-weighted code
([google-research/spherical-cnn](https://github.com/google-research/spherical-cnn),
`spin_spherical_harmonics.py`) uses `[L+1, 2L+1]` indexed $(l, m)$, with
$c_{lm}$ at `[l, L + m]`, and puts spin on its own axis:
`[L+1, 2L+1, n_spins, channels]`.

```
m =     -2    -1    0    1    2
l = 0 [  0     0   c00   0    0  ]
l = 1 [  0   c1-1  c10  c11   0  ]
l = 2 [ c2-2 c2-1  c20  c21  c22 ]
```

Just over half of the $(L+1)(2L+1)$ cells hold data.

**We use the flat layout.** Every spectrum that crosses a module boundary is
`[batch, channels, (L+1)**2]`. The reasons:

- Truncating to $L' < L$, which is how pooling and `resize` band-limit, is a
  prefix slice `[..., :(L'+1)**2]`. In the padded layout it is two slices,
  because the centre of the $m$ axis moves with $L$.
- The only padding is the spin-1 dummy described below. In the padded layout
  anything that does not map 0 to 0 (a bias, the shift in a norm, a pointwise
  nonlinearity) writes into the $|m| > l$ cells. Synthesis then hides it,
  because the Wigner block is zero there too, but norms, per-order power and
  losses computed on the padded tensor would count it.
- It halves the bytes, although S² spectra are small next to the spatial
  grids and the $SO(3)$ blocks, so this is the weakest reason.

The cost is that per-order operations need an index: a weight that is
constant across $m$ must be expanded by $2l+1$ (`repeat_interleave` or a
precomputed index) rather than broadcast along an axis.

**The padded layout is used inside the S² transforms, and only there.**
`S2FFT` and `S2iFFT` contract against a dense Wigner block
(`precomp.rectangular`, `[2L+1, L+1, n_beta]`), so that the loop over orders
becomes one batched matmul. They convert at their boundaries with
`precomp.Rect.to_flat` and `from_flat`. That block is indexed $(m, l)$,
**transposed relative to Esteves'** $(l, m)$, with $m$ at `L + m`. The
padding roughly doubles the table, which is paid once per layer, not per
sample.

Spin 1 has no $l = 0$ term, so its natural length is one shorter, with order
$l$ at `(l-1)*(l+1) : l*(l+2)`. Both spins are nevertheless stored at the
spin-0 length, with **a dummy zero as the first coefficient** when $n = 1$.
Tensors then stay homogeneous when a layer carries both spins at the cost of
one wasted coefficient per channel. Two consequences:

- A normalisation over the coefficient axis must skip that dummy entry.
- Which channels are spin 1 is a convention, not a shape. Esteves' separate
  spin axis rules out mixing spins by construction. Here it is the layer's
  job: a dense layer across all channels would combine them and break
  equivariance.

Converting to or from Esteves' layout is a transpose plus `Rect`. Matching the
layout does not mean matching the values: the sign of $m$, the
Condon–Shortley phase, the spin sign and the transform normalisation all need
checking separately.

## Spectra on $SO(3)$

A general $SO(3)$ spectrum is a **list**, one entry per order, where entry $l$
has shape `[batch, channels, 2l+1, 2l+1]` indexed $(m, n)$. Truncating the list
band-limits the signal, which is how the group path pools
(`modules/equivariant.resize`).

`SO3_to_S2` takes column $n$ of every block and returns the flat $S^2$ layout
above; `S2_to_SO3` pads the other way.

## Grids

Spatial signals are sampled on `n_alpha` longitudes and `n_beta` latitudes,
excluding the poles, with $\beta_j = \pi (j+1) / (n_\beta+1)$
(`SO3lib.from_index_to_beta`). A layer band-limited at $L$ uses
$n_\alpha = n_\gamma = 2(L+1)$. An $SO(3)$ spatial signal carries a third axis:
`[batch, channels, alpha, beta, gamma]`.

The quadrature over $\beta$ is a sine-weighted rectangle rule with weight
$\frac{\pi}{n_\beta+1}\sin\beta$. It is **not** exact at $L = n_\beta$, the
shipped setting; three strict xfails in `tests/test_transforms.py` hold that
open. The rule lives in `modules/quadrature.py` and the analysis transforms
take a `quadrature=` argument, so a replacement is registered there rather than
edited into `SO3fft.py`.

## Transform normalisation

Analysis and synthesis are inverse up to quadrature error, with $\alpha$ and
$\gamma$ following `torch.fft` conventions: `norm="ortho"` over $\alpha$ in both
directions; over $\gamma$, `SO3iFFT` leaves the transform unnormalised and
`SO3FFT` uses `norm="forward"`, so that `SpatialSmoothing`'s average over
$\gamma$ returns the coefficient itself.

One consequence is worth knowing before comparing paths: `SO3iFFT` → smoothing
→ `S2FFT` is scaled by $1/\sqrt{n_\gamma}$ relative to the pure $S^2$ path,
because the first normalises a 2-D ortho FFT and the second a 1-D one, and
$n_\gamma = 2(L+1)$ varies per layer.

## Wigner tables

$d^l_{mn}(\beta)$ is real, and is stored as such — real fp32, not complex64,
which halves both the bytes and the multiply count (`SO3lib.contract` threads
the (re, im) axis through the contraction). One table per order:

| table | file | contents |
| ----- | ---- | -------- |
| full | `FULL_WIGNER_<n_beta>_<l_max>.list` | `[n_beta, 2l+1, 2l+1]` per $l$ |
| spin 0 | `SLICE_WIGNER_N0_<n_beta>_<l_max>.list` | column $n = 0$ |
| spin 1 | `SLICE_WIGNER_N1_<n_beta>_<l_max>.list` | column $n = 1$, from $l = 1$ |

The slices are what the $S^2$ transforms need; only the group path reads the
full table. Each table is packed into **one** buffer (`precomp.Packed`), with
`tables[l]` a view into it: the orders used to be $L+1$ separate buffers
rebuilt into a Python list on every call, and the ragged full table used
`torch.nested`, a prototype API that blocks `torch.compile`. Buffers move with
`.to(device)` but stay out of the `state_dict` (`persistent=False`).
`modules/precomp.py` also accepts the older `<kind>_<n_beta>.list` names, and
converts a complex table on load.
