# S3CNN: Spin-weighted Smoothing Spherical CNN

SO(3)-equivariant convolutional networks on the sphere. One equivariant layer
expresses three architectures — the sCNN of Cohen et al. (2018), the
spin-weighted swsCNN of Esteves et al. (2020, 2023), and the spin-weighted
smoothing S3CNN proposed here — which differ only in where signal equivariance
is imposed. Applied to ERA5 weather forecasting and to Spherical MNIST.

## Quick start

```
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
python generate_precomps.py --n_beta 179 --l_max 179 --compute_wigner
./reproduce.sh wind->wind S3CNNUNet 1 paper
```

The ERA5 dataset is downloaded separately, from
[here](https://universityofbergen-my.sharepoint.com/:f:/g/personal/francesco_ballerin_uib_no1/Epd3GwgY0s5Hp9Pij9lAhxsBEoJDfrdwgzoVC58_ojGl8g?e=5UvH2h);
Spherical MNIST is fetched on first use. Full instructions:
[docs/running.md](docs/running.md).

## Documentation

| file | contents |
| ---- | -------- |
| [docs/overview.md](docs/overview.md) | the problem, the layer, and the one choice the three architectures differ by |
| [docs/notation.md](docs/notation.md) | symbols and formulas |
| [docs/conventions.md](docs/conventions.md) | how signals, spectra and tables are laid out in tensors |
| [docs/transforms.md](docs/transforms.md) | the transform modules, their normalisation, and the quadrature |
| [docs/architectures.md](docs/architectures.md) | `scnn`, `s3cnn`, `swscnn`; the topologies and the model zoo |
| [docs/fused.md](docs/fused.md) | the slab-by-slab SO(3) path and its Triton kernels |
| [docs/running.md](docs/running.md) | install, precomputation, datasets, training, testing, benchmarks |

## Rotation equivariance

Both an swsCNN and an S3CNN are SO(3)-equivariant up to numerical error, while a classical CNN on a lat/lon grid is not (except for rotations about the z-axis). The table below compares a classical CNN and an S3CNN on the wind-prediction task: both models are trained *without* rotation augmentation and then evaluated on an input rotated by Y(π/4). The S3CNN prediction is unchanged (up to numerical error) after rotating back to the original frame, while the CNN prediction is not.

<div align="center">
<table>
  <tr>
    <th>Model</th>
    <th>Ground truth</th>
    <th>Predicted, β=0</th>
    <th>Predicted, β=π/4</th>
    <th>Rotation error</th>
  </tr>
  <tr>
    <td>CNN</td>
    <td><img src="figures/rescaled_NT_ground_truth.png" width="130"></td>
    <td><img src="figures/rescaled_NT_cnn_NR.png" width="130"></td>
    <td><img src="figures/rescaled_NT_cnn_R.png" width="130"></td>
    <td><img src="figures/rescaled_NT_cnn_error.png" width="130"></td>
  </tr>
  <tr>
    <td>S3CNN</td>
    <td><img src="figures/rescaled_NT_ground_truth.png" width="130"></td>
    <td><img src="figures/rescaled_NT_so3_NR.png" width="130"></td>
    <td><img src="figures/rescaled_NT_so3_R.png" width="130"></td>
    <td><img src="figures/rescaled_NT_so3_error.png" width="130"></td>
  </tr>
</table>
</div>

<p align="center"><i>0 km/h</i> &nbsp;<img src="figures/jet_colormap.png" height="14"> &nbsp;<i>28 km/h</i></p>


Only the magnitude of the predicted wind vector field is shown. The rotation error (last column) is the difference between the β=0 and β=π/4 predictions, brought back to a common reference frame, which is negligible for the S3CNN, but large for the CNN.

## Plots

The scripts in `plotting/` generate the figures; fill in the run ids from the
logs you produced.

## Citing

See `CITATION.cff`.

## Credits

- Francesco Ballerin (Universitetet i Bergen - Department of Mathematics)
- Erlend Grong (Universitetet i Bergen - Department of Mathematics)
- Nello Blaser (Universitetet i Bergen - Department of Informatics)

For any question: francesco@ballerin.it .
