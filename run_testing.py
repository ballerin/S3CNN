import os
import pickle

import torch

from SphericalDataset import WeatherDataset, SphericalMNISTDataset
from utils.data_utils import make_dataloader
from utils.params import check_keys

from modules import zoo
from modules.losses import S2RMSELoss

from trainer_tester import Tester

ALLOWED_ROTATIONS = ["no", "beta", "full"]
ALLOWED_FFT_METHODS = ["full",] #Delta is not currently supported after refactor

TEST_YEARS = ["2022", "2023"]

#: Prefix of the files a test run writes, by rotation.
ROTATION_TAG = {"no": "NR", "beta": "RB", "full": "RF"}


def make_test_dataset(task: zoo.Task, rotate: str, device):
    """The test set, rotated once by a fixed list when `rotate` asks for it."""
    if rotate == "full":
        rotations_list = pickle.load(open("precomp/FULL_TEST_ROTATIONS.list", "rb"))
    elif rotate == "beta":
        rotations_list = pickle.load(open("precomp/BETA_TEST_ROTATIONS.list", "rb"))
    else:
        rotations_list = None

    if task.data == "weather":
        test_dataset = WeatherDataset(root=task.root, folders=TEST_YEARS, train=False,
                                      in_datatype=task.x, out_datatype=task.y)
        test_dataset.preload()
        if rotate != "no":
            test_dataset.rotate_from_list(device=device, rotations_list=rotations_list,
                                          rotation_type=rotate)
    else:
        # With caching on, the dataset applies the fixed rotation list itself.
        test_dataset = SphericalMNISTDataset(split="test", mode="spherical",
                                             n_alpha=task.n_alpha, n_beta=task.n_beta,
                                             in_datatype=task.x, out_datatype=task.y,
                                             rotation_type=rotate, rotations_list=rotations_list,
                                             compute_device=device,
                                             caching=True, verbose=False)
    return test_dataset


def run_testing(gpu_id: int, # gpu id to use
              params: dict = None, # dictionary of parameters
              model_type: str = None, # id of the model to use
              experiment: str = None,
              batch_size: int = 1,
              path: str = None,
              rotate: str = "no",
              fft_method: str = "full",
              use_batch_norm: bool = False,
              use_layer_norm: bool = False,
              nonlinearity: str = None,
              ):
    """
    Test the best checkpoint of a training run.

    The model is built from the same `modules.zoo` entry `run_training` used,
    so the architectural keys below must match the ones the run was trained
    with; `load_state_dict` is strict and says so when they do not.

    Parameters:
    - gpu_id: int, id of the GPU to use
    - params: dict, dictionary containing all parameters (if provided, overrides other parameters)
    - model_type: str, name of the model; see `run_training` for the naming
    - experiment: str, name of the experiment; one of `zoo.EXPERIMENTS`
    - batch_size: int, size of the test batches
    - path: str, the training run's directory; weights are read from
      `snapshots/checkpoint_best.pt` and results written next to it
    - rotate: str, rotation applied to the test set: "no", "beta" or "full"
    - fft_method: str, method to use for FFT; only "full" is implemented
    - use_batch_norm: bool, as in training
    - use_layer_norm: bool, as in training
    - nonlinearity: str, as in training; None is the architecture's own
    """

    if params is not None:
        check_keys(params, run_testing)

        experiment = params['experiment']
        model_type = params['model_type']
        batch_size = params['batch_size']
        rotate = params['rotate']
        path = params['path']

        fft_method = params.get('fft_method', 'full')
        nonlinearity = params.get('nonlinearity', None)  # None: the architecture's own
        use_batch_norm = params.get('use_batch_norm', False)
        use_layer_norm = params.get('use_layer_norm', False)

    # Parameters check
    spec = zoo.spec(experiment, model_type)
    task = zoo.task(experiment)
    if rotate not in ALLOWED_ROTATIONS:
        raise ValueError(f"Unknown rotation type: {rotate}.")
    if fft_method is None:
        fft_method = "full"
    if fft_method not in ALLOWED_FFT_METHODS:
        raise ValueError(f"Unknown FFT method: {fft_method}.")
    if use_batch_norm and use_layer_norm:
        raise ValueError(f"Cannot use BOTH batch normalization and layer normalization.")

    DEVICE = torch.device("cuda", gpu_id)

    weights_path = os.path.join(path, "snapshots/checkpoint_best.pt")
    tag = ROTATION_TAG[rotate]
    log_path = os.path.join(path, f"{tag}_testing_log.txt")
    losses_path = os.path.join(path, f"{tag}_test_losses.pckl")
    accuracies_path = os.path.join(path, f"{tag}_test_accuracies.pckl")

    model = zoo.build(experiment, model_type, nonlinearity=nonlinearity,
                      use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                      fft_method=fft_method).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=DEVICE))

    test_dataset = make_test_dataset(task, rotate, DEVICE)
    test_dataloader = make_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    # Classification is scored by accuracy alone.
    lossfnc = None if task.classification else S2RMSELoss(n_beta=task.n_beta)

    with open(log_path,'w') as ff:
            ff.write(f"-- BEGIN OF LOG --\n")

    tester = Tester(model = model,
                    in_datatype = task.in_datatype,
                    out_datatype = task.out_datatype,
                    dataloader = test_dataloader,
                    lossfnc = lossfnc,
                    gpu_id = gpu_id,
                    fft_method = fft_method,
                    log_path = log_path,
                    losses_path = losses_path,
                    accuracies_path = accuracies_path,
                    autoencoding = task.autoencoding,
                    rotate = rotate,
                    train_dataloader = None,
                    )

    tester.test()

    with open(log_path,'a') as ff:
            ff.write(f"-- END OF LOG --\n")
