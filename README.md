# Expressive SO(3)-equivariant Neural Networks

Before running any script:

* install the dependencies (also in `requirements.txt`)
* run `generate_precomps.py` to precompute Wigner (small) d-matrices and test rotations
* download the dataset (or part of) from [here](https://universityofbergen-my.sharepoint.com/:f:/g/personal/francesco_ballerin_uib_no1/Epd3GwgY0s5Hp9Pij9lAhxsBEoJDfrdwgzoVC58_ojGl8g?e=5UvH2h)

## Precomputations
The precomputed files are stored in `precomp/` and are the coefficients used for FFT/iFFT as well as generating the random rotations performed during test.
These are:
```
precomp/SLICE_WIGNER_N0_<n_beta>_<MAX_L>.list
precomp/SLICE_WIGNER_N1_<n_beta>_<MAX_L>.list
precomp/FULL_WIGNER_<n_beta>_<MAX_L>.list
precomp/BDELTAS_<n_beta>_<MAX_L>.list
precomp/CONJ_PROD_<n_beta>_<MAX_L>.list
precomp/SINES_<n_beta>.torch
precomp/ROTATIONS_<N_ROTATIONS>.pckl
```
The files `precomp/SLICE_WIGNER_N0_<n_beta>_<MAX_L>.list` and `precomp/SLICE_WIGNER_N1_<n_beta>_<MAX_L>.list` contain a subset of the full Wigner d-matrices from $L=0$ to $L=$ MAX_L (included), considering only the case column corresponding to $N=0$ and $n=1$ respectively. The angle beta is discretized according to `n_beta`, excluding the poles.

The file `precomp/FULL_WIGNER_<n_beta>_<MAX_L>.list` contains all the Wigner d-matrices from L=0 to L=MAX_L discretized for `n_beta` equidistant angles from 0 to $\pi$ excluding the poles, indexed by $m,n\in\mathbb R$, $-L\leq m,n \leq L$.

The file `precomp/BDELTAS.list` contains the Wigner d-matrices from L=0 to L=MAX_L considering only $\beta=\frac{\pi}{2}$ and with the $\Delta$ formalism as in the paper.

The file `precomp/CONJ_PROD.list` consists of the precomputed products of Delta matrices (BDELTAS.list).

The file `precomp/SINES.list` consists of a precomputed mash of the sine function used for the Delta FFT method.

The file `precomp/test_rotations.pckl` contains a list of randomly generated rotation angles used during test.

## Experiment types

| key                          | Description                                 
| ---------------------------- | -------------------------------------------------- |
| `wind->wind`                 | ERA5 wind (vector) to T+24h wind (vector)          |
| `temp->wind`                 | ERA5 temp (scalar) to T+0h wind (vector)           |
| `autoencoder`                | ERA5 wind (vector) compression to max 512 float32  |
| `MNIST-scalar-classification`| Classification of Spherical MNIST                  |
| `MNIST-vf-classification`    | Classification of Spherical MNIST with Sobel filt. |
| `MNIST-scalar-to-vector`     | Spherical MNIST to Spherical Sobel MNIST           |
| `MNIST-vector-to-scalar`     | Spherical Sobel MNIST to Spherical MNIST           |

## Models

We have implemented the following models

| key                          | Description                                 
| --------------------- | --------------------------------------------------------- |
| `SO3UNet`             | UNet architecture with full-SO(3) equiv. layers           |
| `SpinsSO3UNet`        | UNet architecture with spin-weighted equiv. layers        |
| `CNNUNet`             | CNN UNet architecture                                     |
| `SO3autoencoder`      | Autoencoder architecture with full-SO(3) equiv layers     |
| `SpinsSO3autoencoder` | Autoencoder architecture with spin-weighted equiv. layers |
| `CNNautoencoder`      | Autoencoder architecture based on CNN                     |
| `SO3MNIST`            | Equivariant classifier based on full-SO(3) layers         |
| `SpinsSO3MNIST`       | Equivariant classifier based on spin-weighted layers      |
| `CNNMNIST`            | Classifier based on conv. layers                          |

### SO(3)-Equivariant models

| key           | Value             | Description                                 
| ------------- |-------------------| ------------------------------------------------------------ |
| `layer_type`  | `so3` or `spins`  | `so3` for full layers, `spins` for spin-weighted             |
| `fft_method`  | `full` or `delta` | Only `full` is currently implemented. Fast but not sp.-eff.  |
| `n_alpha`     |  int              | Longitude resolution (usually 120 or 360)                    |
| `n_beta`      |  int              | Latitude resolution excluding poles (usually 59 or 179)      |
| `resolutions` |  list of ints     | List of massimum order of Fourier coefficients at each layer |
| `widths`      |  list of ints     | List of channels at each layer                               |
| `spins`       |  list of lists    | List of spins (0,1) at each layer. Both= half channels each  |

```
model = SO3UNet(layer_style="so3",
                fft_method="full",
                n_alpha = 360, n_beta=179,
                resolutions = [179, 100, 50, 100, 179],
                widths = [1, 2, 8, 30, 8, 1],
                spins = [[0,], [0,1], [0,1], [0,1], [0,1], [1,]],
                )
```

### CNN Models
| key           | Value             | Description                                 
| ------------- |---------------------------------------| ---------------------------------------- |
| `in_datatype` | `complex-scalar` or `complex-vector`  | Input type |
| `n_alpha`     |  int              | Longitude resolution (usually 120 or 360)                    |
| `n_beta`      |  int              | Latitude resolution excluding poles (usually 59 or 179)      |
| `widths`      |  list of ints     | List of channels at each layer                               |

```
model = CNNMNISTClassifier(in_datatype="complex-scalar",
                           n_alpha=120, n_beta=59,
                           widths=[1, 8, 32, 128, 256, 512])
```

## Dataset

Datasets are Spherical and have a common class.

### ERA5 Dataset
The dataset needs to be retrieved and is stored in `dataset/` and is organized by year. Example:
```
dataset
├── type (e.g. wind->wind, temp->wind, ...)
    ├── 2000
    │   ├── X
    │   │   ├── 0.npy
    │   │   ├── 1.npy
    │   │   ├── ...
    │   │   └── 8735.npy
    │   └── Y
    │       ├── 0.npy
    │       ├── 1.npy
    │       ├── ...
    │       └── 8735.npy
    ├── 2001
    │   ├── X
    │   │   └── ...
    │   └── Y
    │       └── ...
    ├── ...
    └── 2020
        ├── X
        │   └── ...
        └── Y
            └── ...
```
`type` can be a folder or nested folders, and is used to differentiate between different types of datasets (scalar to vector, vector to vector...) as well as the coarseness of the dataset (lat/lon grid or coarseness in time).

```
WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
               folders=train_folders,
               train=True,
               in_datatype="vector", out_datatype="vector")
```

### Spherical MNIST Dataset

The data is not saved locally but retrieved on first runtime through the the Pytorch module when calling the `SphericalMNISTDataset` class the first time.

```
SphericalMNISTDataset(split="train", mode="equirectangular",
                      n_alpha=120, n_beta=59,
                      in_datatype="scalar", out_datatype="label")
```

## Logs

Training logs and figures are saved under `experiments/`, divided by experiment type (wind->wind, temp->wind, autoencoder, ...). If any error arises, initialize the empty folders.

## Training / Testing

To train the models simply schedule the experiments to be executed in the `train_schedule/schedule.py` and execute the file ```run_all_experiments.py``` with the desired GPUs. Execution proceeds in parallel w.r.t. the GPUs.

Example: 
```
$ python run_all_experiments.py --GPU 0
$ python run_all_experiments.py --GPU 2,3,5
```

A train schedule job and will result in a log being generatered which also contains the weights of the best performing iteration. At starttime an id will be assigned to the job. It has the following keys:
| key             | Values                                 | Default   |
| --------------- | -------------------------------------- | --------- |
| `model_type`    | Name of the model (string)             | None      |
| `experiment`    | `"wind->wind"`, `"temp->wind"`, ...    | None      |
| `total_epochs`  | Any positive integer                   | None      |
| `early_stopping`| Any positive integer                   | None      |
| `save_every`    | Any positive integer                   | None      |
| `batch_size`    | Any positive integer                   | None      |
| `fft_method`    | `"delta"`, `"full"` (not used for CNN) | `"full"`  |
| `dataset_size`  | Any positive integer                   | None      |
| `rotate`        | `"no"`, `"random"`                     | None      |
| `log_id`        | String                                 | None      |

A test schedule job requires the corresponding training job to have been executed first. It requires the following keys:

| key             | Values                                 | Default   |
| --------------- | -------------------------------------- | --------- |
| `model_type`    | Name of the model (string)             | None      |
| `experiment`    | `"wind->wind"`, `"temp->wind"`, ...    | None      |
| `path`          | Relative path of the training folder   | None      |
| `batch_size`    | Any positive integer                   | None      |
| `fft_method`    | `"delta"`, `"full"` (not used for CNN) | `"full"`  |
| `rotate`        | `"no"`, `"beta"`, `"full"`             | None      |



## Plots

To generate the plots use the files `plot_windwind.py`, `plot_tempwind.py`, and `plot_autoencoder.py`. The files have to be filled in with the ids from the logs that have been run.

## Credits

- Francesco Ballerin (Universitetet i Bergen - Department of Mathematics)
- Erlend Grong (Universitetet i Bergen - Department of Mathematics)
- Nello Blaser (Universitetet i Bergen - Department of Informatics)

For any question: francesco.ballerin@uib.no .