# Notation

Symbols and formulas as the code uses them. Derivations: the paper, and
Kostelec & Rockmore 2008 for the transform.

## Symbols

| symbol | meaning |
| ------ | ------- |
| $\alpha, \beta, \gamma$ | ZYZ Euler angles: $A = Z(\alpha)\,Y(\beta)\,Z(\gamma)$ |
| $L$ | spectral band-limit of a function/layer |
| $l$ | degree (order) of a representation, $0 \le l \le L$ |
| $m, n$ | row and column index within degree $l$, $-l \le m, n \le l$ |
| $n$ | is also the **spin weight** of a signal, $0$ or $1$ |
| $d^l_{mn}(\beta)$ | Wigner (small) $d$-matrix |
| $D^l_{mn}$ | Wigner $D$-matrix|
| $\hat f^l_{mn}$ | spectral coefficient |
| $\psi$ | filter |
| $n_\alpha, n_\beta, n_\gamma$ | grid resolutions |

## Wigner functions

$$D^l_{mn}(\alpha, \beta, \gamma) = e^{-im\alpha}\, d^l_{mn}(\beta)\, e^{-in\gamma}$$

with $d^l$ real.

## Fourier transforms on $S^2$ / $SO(3)$

Fourier transforms of a band-limited $f$ on the group:

$$f(\alpha,\beta,\gamma) = \sum_{l=0}^{L} \sum_{|m|,|n| \le l} \hat f^l_{mn}\, D^l_{mn}(\alpha,\beta,\gamma)$$

$$\hat f^l_{mn} = \frac{2l+1}{8\pi^2} \int_{SO(3)} f(A)\, \overline{D^l_{mn}(A)}\; dA$$

## Spin weight, and $S^2$ as a slice of $SO(3)$

A signal is $n$-**equivariant** when

$$f(A\,Z(\gamma)) = f(A)\, e^{-in\gamma},$$

which whenever is true makes $f$ a function of $\alpha$ and $\beta$ alone, i.e. a field on $S^2$: $n = 0$ is a scalar field, $n = 1$ a vector field carried as one complex
number per point. Such an $f$ has spectral support in column $n$ only, so its
spectrum is the vector $\hat f^l_{m}$ := $\hat f^l_{mn}$. This is why an $S^2$
signal is stored as one column of an $SO(3)$ spectrum, and why the $\gamma$
integral in the smoothing operator collapses.

## Convolution

On a compact group, correlation is a per-degree product of spectra. The three
filter classes are exactly the three ways of restricting that product:

| layer | filter | contraction | input | output |
| ----- | ------ | ----------- | ------ | ------ |
| `SO3Conv` | $\psi^l_n$ | $\hat y^l_{mn} = \sum_i \hat\psi^l_{ion}\, \hat x^l_{im}$ | $S^2$ | $SO(3)$, lifting |
| `SO3toSO3Conv` | $\psi^l_{m,n}$ | $\hat y^l_{mp} = \sum_{i,k} \hat\psi^l_{iokp}\, \hat x^l_{imk}$ | $SO(3)$ | $SO(3)$ |
| `SpinSO3Conv` | $\psi^l$ | $\hat y^l_{m} = \sum_i \hat\psi^l_{io}\, \hat x^l_{im}$ | $S^2$ | $S^2$ |

$i$ and $o$ are input and output channels.

## The smoothing operator

This is what projects $SO(3)$ back to $S^2$ after the nonlinearity in `s3cnn`
and in the spatial domain is the mean over $\gamma$ against the spin
character,

$$(\mathcal{S}f)(\alpha, \beta) = \frac{1}{2\pi}\int_0^{2\pi} f(\alpha, \beta, \gamma)\, e^{in\gamma}\; d\gamma,$$

as a discrete average over the $n_\gamma$ samples.

It can also be computed in the spectral domain by zeroing out all columns of the spectrum except $n$.
