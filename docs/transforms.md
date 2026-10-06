# Transforms

`modules/SO3fft.py`. Four modules, one per direction and domain. Algorithm:
Kostelec & Rockmore 2008, sec. 3.

| module | in | out |
| ------ | -- | --- |
| `S2FFT` | `[B, C, n_alpha, n_beta]` | `[B, C, (L+1)**2]` |
| `S2iFFT` | `[B, C, (L+1)**2]` | `[B, C, n_alpha, n_beta]` |
| `SO3FFT` | `[B, C, n_alpha, n_beta, n_gamma]` | list of `[B, C, 2l+1, 2l+1]` |
| `SO3iFFT` | list of `[B, C, 2l+1, 2l+1]` | `[B, C, n_alpha, n_beta, n_gamma]` |

`S2_to_SO3` and `SO3_to_S2` move between the two spectral layouts without
touching the spatial domain: one pads a column into full blocks, the other
reads column $n$ back out.

## The three axes are not alike

An $SO(3)$ synthesis is

$$\hat f^l_{mn} \;\longrightarrow\; g[m, \beta, n] \;\longrightarrow\; \text{FFT over } \alpha \;\longrightarrow\; \text{FFT over } \gamma$$

where the first step is the $\beta$-synthesis,

$$g[m, \beta, n] \;=\; \sum_{l \,\ge\, \max(|m|,|n|)} \hat f^l_{mn}\; d^l_{mn}(\beta).$$

It sums over $l$, not over a frequency conjugate to a coordinate; its kernel is
a Wigner $d$, not a complex exponential; and there is no fast transform for it,
so it is a dense contraction against the precomputed table, $O(L)$ work per
output cell. The $\alpha$ and $\gamma$ steps are ordinary FFTs and cost
essentially nothing by comparison.

That is why the $\beta$ step, and only the $\beta$ step, gets a hand-written
kernel ([fused.md](fused.md)), and why $\beta$ is the axis the fused path cuts
into slabs — a single $\gamma$ needs every coefficient, so a $\gamma$ slab
would still require the whole grid.

For an $n$-equivariant signal the $\gamma$ integral collapses by definition,
which is what reduces `SO3FFT` to `S2FFT`: an FFT over $\alpha$ and the
quadrature against the column $d^l_{mn}$ over $\beta$.

## Normalisation

Inverse up to quadrature error. The exact scaling, and the $1/\sqrt{n_\gamma}$
offset between the group path and the pure $S^2$ path, is in
[conventions.md](conventions.md#transform-normalisation). `SO3iFFT` writes at
the `fftshift`ed position directly rather than rolling afterwards, which would
be another full copy of the grid.

Only `method="full"` is implemented. `method="delta"` — the $\Delta$-matrix
route of the paper, which trades the full table for
$\Delta^l = d^l(\pi/2)$ and precomputed products — is present as a stub, and
`generate_precomps.py` still writes the `BDELTAS`/`CONJ_PROD`/`SINES` files it
would need.

## Quadrature over $\beta$

`modules/quadrature.py`. Analysis integrates against $d^l_{mn}(\beta)$ over the
latitudes; a rule owns the nodes and the complete analysis table

$$A^l_{b,m} \;=\; \text{measure} \cdot \text{order\_scale}(l) \cdot \text{node\_weights}(b) \cdot d^l_{b,m}.$$

The shipped rule, `RectangleRule`, is the sine-weighted rectangle rule on the
equiangular nodes: measure $\pi/(n_\beta+1)$, order scale $(2l+1)/2$, node
weights $\sin\beta_b$. It is **rank-deficient at $L = n_\beta$**, which is what
the models are configured with — the analysis–synthesis roundtrip loses 21–33%
there, and three strict xfails in `tests/test_transforms.py` hold that open.
The rule is second order otherwise: doubling $n_\beta$ divides the roundtrip
error by four.

The module exists so that replacing the rule is a swap, not surgery. `S2FFT`
and `SO3FFT` take `quadrature=`; a rule that factorises declares
`diagonal = True` and the transforms then contract the pieces without ever
materialising $A$, which on the $SO(3)$ path is the size of the whole Wigner
table. Candidates: a pseudo-inverse analysis operator (not diagonal, hence the
`operator()` method), $\beta$-oversampling (keeps the nodes), and
Gauss-Legendre (moves them, so the ERA5 data would need resampling).
