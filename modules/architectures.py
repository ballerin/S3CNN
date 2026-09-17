"""The three architectures, as configuration rather than as code paths.

They differ in one choice -- *where* signal-equivariance is imposed -- and that
choice fixes both the filter class and which nonlinearities are admissible:

| arch     | equivariance      | filter                     | sigma            | after |
| -------- | ----------------- | -------------------------- | ---------------- | ----- |
| `scnn`   | at the output     | psi^l_{m,n}, O(L^3)        | unconstrained    | Cohen et al. 2018 |
| `s3cnn`  | after every sigma | psi^l_n, O(L^2)            | unconstrained    | ours |
| `swscnn` | inside the filter | psi^l, O(L)                | spin-preserving  | Esteves et al. 2020, 2023 |

`scnn` and `s3cnn` both lift to SO(3); they part company over what happens
after the nonlinearity, where `s3cnn` projects back to S2 every layer and
`scnn` stays on the group until the output.  `swscnn` never leaves S2, so the
nonlinearity has to preserve the spin weight by itself -- which is why the
first two admit a ReLU and the third does not.

For an input in X_p the filter coefficients psi^l_{s,n} contribute only at
s = p, so the first layer of an `scnn` and the first layer of an `s3cnn` are
the same operator; the architectures diverge only from layer two onward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    """One row of the table above."""

    name: str
    #: Filter class: the full group filter, one coefficient per (l, n), or one per l.
    filters: str            # "so3" | "spin" | "degree"
    #: Whether spatial smoothing collapses SO(3) back to S2 after every sigma.
    project: str            # "per_layer" | "never"
    #: Nonlinearities this architecture admits, canonical choice first.
    sigmas: tuple[str, ...]
    reference: str

    @property
    def lifts(self) -> bool:
        """Does the signal live on SO(3) between layers?"""
        return self.filters in ("so3", "spin")

    @property
    def projects(self) -> bool:
        return self.project == "per_layer"

    @property
    def sigma(self) -> str:
        """The architecture's canonical nonlinearity."""
        return self.sigmas[0]

    def check_sigma(self, sigma: str) -> str:
        if sigma not in self.sigmas:
            raise ValueError(
                f"{self.name} admits {list(self.sigmas)}, got {sigma!r}. "
                f"Equivariance is imposed {ORIGIN[self.name]}, which is what "
                f"constrains the nonlinearity.")
        return sigma


ORIGIN = {"scnn": "at the output only",
          "s3cnn": "after every nonlinearity",
          "swscnn": "inside every filter"}

ARCHITECTURES = {
    "scnn": Architecture(
        name="scnn", filters="so3", project="never",
        sigmas=("leaky_relu", "relu"), reference="Cohen et al. 2018"),
    "s3cnn": Architecture(
        name="s3cnn", filters="spin", project="per_layer",
        sigmas=("leaky_relu", "relu"), reference="ours"),
    "swscnn": Architecture(
        name="swscnn", filters="degree", project="never",
        sigmas=("magnitude", "phase_collapse"),
        reference="Esteves et al. 2020, 2023"),
}

#: What the same three architectures used to be called.  Kept so that
#: checkpoints and schedules written against the old vocabulary still resolve;
#: `full_so3` in particular was the sCNN path, not ours.
LEGACY_LAYER_STYLES = {"full_so3": "scnn", "so3": "s3cnn", "spins": "swscnn"}


def resolve(arch: str | None = None, layer_style: str | None = None) -> Architecture:
    """The `Architecture` named by `arch`, or by the older `layer_style`."""
    if arch is not None and layer_style is not None:
        raise ValueError("Pass arch or layer_style, not both; "
                         f"layer_style={layer_style!r} is the old name for "
                         f"{LEGACY_LAYER_STYLES.get(layer_style, '?')!r}.")
    if arch is None:
        if layer_style is None:
            raise ValueError(f"An architecture is required: {list(ARCHITECTURES)}")
        if layer_style not in LEGACY_LAYER_STYLES:
            raise ValueError(f"Unknown layer_style {layer_style!r}; "
                             f"expected one of {list(LEGACY_LAYER_STYLES)}. "
                             f"Prefer arch={list(ARCHITECTURES)}.")
        arch = LEGACY_LAYER_STYLES[layer_style]
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown arch {arch!r}; expected one of {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch]
