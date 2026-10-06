import os
import pickle
import time

import torch

from training_utils import plot_data

from SphericalDataset import WeatherDataset, SphericalMNISTDataset
from utils.data_utils import make_dataloader
from utils.params import check_keys

from modules import zoo
from modules.losses import S2MSELoss

from trainer_tester import Trainer

ALLOWED_ROTATIONS = ["no", "offset", "beta", "full"]
ALLOWED_FFT_METHODS = ["full",] #Delta is not currently supported after refactor

#: Training years start here; `dataset_size` counts them.
FIRST_TRAIN_YEAR = 2000
VAL_YEARS = ["2020", "2021"]


def get_experiment_path(experiment:str, mode:str, rotate:str, dataset_size:int, log_id:str|int):
    log =  "experiments/"+experiment+"/"+mode+"/"
    if rotate in ["no", "offset"]:
        log += "NR/"
    elif rotate == "full":
        log += "RF/"
    elif rotate == "beta":
        log += "RB/"
    else:
        raise ValueError(f"Unknown rotation type: {rotate}")
    if dataset_size is not None and dataset_size >= 1:
        log += str(dataset_size)+"/"
    log += str(log_id)+"/"
    return log


def make_datasets(task: zoo.Task, rotate: str, dataset_size: int, device,
                  preload_dataset: bool, verbose: bool):
    """Train and validation sets; validation is rotated once, by a fixed list."""
    if rotate == "full":
        rotations_list = pickle.load(open("precomp/FULL_VALIDATION_ROTATIONS.list", "rb"))
    elif rotate == "beta":
        rotations_list = pickle.load(open("precomp/BETA_VALIDATION_ROTATIONS.list", "rb"))
    else:
        rotations_list = None

    if task.data == "weather":
        if dataset_size is None:
            raise ValueError("Weather experiments need a dataset_size (number of training years).")
        train_folders = [str(year) for year in range(FIRST_TRAIN_YEAR, FIRST_TRAIN_YEAR+dataset_size)]
        train_dataset = WeatherDataset(root=task.root, folders=train_folders, train=True,
                                       in_datatype=task.x, out_datatype=task.y)
        val_dataset = WeatherDataset(root=task.root, folders=VAL_YEARS, train=False,
                                     in_datatype=task.x, out_datatype=task.y)
        train_dataset.preload()
        val_dataset.preload()
        if rotate != "no":
            val_dataset.rotate_from_list(device=device, rotations_list=rotations_list,
                                         rotation_type=rotate)
    else:
        common = dict(mode="spherical", n_alpha=task.n_alpha, n_beta=task.n_beta,
                      in_datatype=task.x, out_datatype=task.y, rotation_type=rotate,
                      compute_device=device, caching=preload_dataset, verbose=verbose)
        train_dataset = SphericalMNISTDataset(split="train", **common)
        val_dataset = SphericalMNISTDataset(split="val", rotations_list=rotations_list, **common)

    return train_dataset, val_dataset


def run_training(gpu_id: int, # gpu id to use
                 params: dict = None,
                 experiment: str = None,
                 model_type: str = None, # id of the model to use
                 nonlinearity: str = None, # type of nonlinearity to use
                 total_epochs: int = 100, # How many epochs to train in total
                 starting_epoch: int = 0,
                 pretrained_weights: str = None, # Path to weights to initialize the model from
                 early_stopping:int = 10,
                 save_every: int = -1, # When to save a complete snapshot of the model
                 batch_size: int = 1,
                 lr: float = 1e-3,
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 recompute: bool = True,
                 fft_method:str = "full", # Only for SO(3) models
                 dataset_size: int = None,
                 rotate: str = "no",
                 log_id: str = None,
                 preload_dataset:bool = True,
                 gradient_accumulation: int = 1,
                 detect_anomaly: bool = False,
                 verbose: bool = False,
                 ):
    """
    Run the training of the model with the given parameters.

    The model for `(experiment, model_type)` and the data it reads come from
    `modules.zoo`, which `run_testing` reads as well.

    Parameters:
    - gpu_id: int, id of the GPU to use
    - params: dict, dictionary containing all parameters (if provided, overrides other parameters)
    - experiment: str, name of the experiment; one of `zoo.EXPERIMENTS`
    - model_type: str, name of the model; one of the pairs in `zoo.ZOO` for this experiment
    - nonlinearity: str, type of nonlinearity to use; None is the architecture's own
    - total_epochs: int, total number of epochs to train
    - starting_epoch: int, epoch number to start training from (useful for resuming)
    - pretrained_weights: str, path to pretrained weights to initialize the model
    - early_stopping: int, number of epochs to wait for improvement before stopping
    - save_every: int, how often to save the model
    - batch_size: int, size of the batches for training
    - lr: float, Adam learning rate. NOT TUNED for the current code: 1e-3 was
      chosen when the convolution kernels were initialised as a real constant.
      Complex Glorot initialisation replaced that and raises the output RMS at
      initialisation 40-200x, and the rate has not been re-tuned since. Sweep
      it before comparing against published numbers, then fix the default here.
    - use_batch_norm: bool, whether to use batch normalization
    - use_layer_norm: bool, whether to use layer normalization
    - recompute: bool, rebuild each SO(3) slab in the backward pass instead of
      keeping it (lifting models only). True, the default, is the memory-lean
      path. False spends memory to save about 1.3x on a training step and gives
      bit-identical gradients, but it is the batch size that decides whether it
      fits: measured on wind->wind S3CNNUNet at 80 GB, batch 8 takes 46.6 GB and
      batch 16 does not fit at all. See the note beside `modules.fused.slab`.
    - fft_method: str, method to use for FFT; only "full" is implemented
    - dataset_size: int, number of training years for weather experiments (1 to 10)
    - rotate: str, rotation augmentation: "no", "offset", "beta" or "full"
    - log_id: str, unique identifier for this run of this experiment (for logging purposes)
    - preload_dataset: bool, whether to preload the whole dataset into RAM memory
    - gradient_accumulation: int, optimiser steps are taken every this many batches
    - detect_anomaly: bool, run autograd's anomaly detector. It records a traceback
      for every forward operation and checks every backward for NaN, which costs
      about 5x the step time, so it is for debugging a specific NaN, not for
      training runs
    - verbose: bool, whether to print detailed logs during training

    Experiments:
    - "wind->wind": Wind speed prediction the following day from wind speed
    - "temp->wind": Wind speed prediction the same day from temperature
    - "autoencoder": Autoencoder for wind speed
    - "MNIST-scalar-classification": MNIST classification with scalar input
    - "MNIST-vf-classification": MNIST classification with vector-field input
    - "MNIST-scalar-to-vector": MNIST scalar to vector prediction
    - "MNIST-vector-to-scalar": MNIST vector to scalar prediction

    Model types are named after the architecture (see `modules/architectures.py`):
    - "S3CNN*": s3cnn, ours -- SO(3) convolution, projected back to S2 after every nonlinearity
    - "SwsCNN*": swscnn, Esteves et al. 2020, 2023 -- spin-weighted filters, never leaves S2
    - "CNN*": planar CNN baseline on the lat/lon grid
    and after the topology: "*UNet", "*Autoencoder", "*MNIST" (classifier), and
    the ablations "*Flat" (no pooling or skips) and "*OnlyOneSpin".
    """

    if params is not None:
        # Everything below reads `params` with `.get`, so a misspelled key would
        # otherwise be a silent default rather than an error.
        check_keys(params, run_training)

        # Required: a schedule entry without one of these is a typo, not a default.
        experiment = params['experiment']
        model_type = params['model_type']
        total_epochs = params['total_epochs']
        batch_size = params['batch_size']
        rotate = params['rotate']

        nonlinearity = params.get('nonlinearity', None)  # None: the architecture's own
        fft_method = params.get('fft_method', 'full')
        early_stopping = params.get('early_stopping', total_epochs)
        starting_epoch = params.get('starting_epoch', 0)
        save_every = params.get('save_every', -1)  # Do not save by default
        pretrained_weights = params.get('pretrained_weights', None)
        lr = params.get('lr', lr)
        use_batch_norm = params.get('use_batch_norm', False)
        use_layer_norm = params.get('use_layer_norm', False)
        recompute = params.get('recompute', True)
        log_id = params.get('log_id', None)
        dataset_size = params.get('dataset_size', None)  # None: not a weather experiment
        preload_dataset = params.get('preload_dataset', preload_dataset)
        verbose = params.get('verbose', verbose)
        gradient_accumulation = max(params.get('gradient_accumulation', 1), 1)
        detect_anomaly = params.get('detect_anomaly', False)

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

    log_id = str(int(time.time())) if log_id is None else str(log_id) #This is the id for the unique run
    logs_path = get_experiment_path(experiment, spec.mode, rotate, dataset_size, log_id)

    DEVICE = torch.device("cuda", gpu_id)

    # The model is built before the data is loaded: a nonlinearity its
    # architecture rejects should fail in seconds, not after the preload.
    model = zoo.build(experiment, model_type, nonlinearity=nonlinearity,
                      use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                      recompute=recompute, fft_method=fft_method)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset, val_dataset = make_datasets(task, rotate, dataset_size, DEVICE,
                                               preload_dataset, verbose)
    train_dataloader = make_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = make_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    if task.classification:
        lossfnc = torch.nn.CrossEntropyLoss()
    else:
        lossfnc = S2MSELoss(n_beta=task.n_beta)

    # Load weights if provided
    if pretrained_weights is not None:
        if os.path.isfile(pretrained_weights):
            print(f"Loading weights from {pretrained_weights}.")
            model = model.to(DEVICE)
            model.load_state_dict(torch.load(pretrained_weights, map_location=DEVICE))
        else:
            raise ValueError(f"Provided pretrained_weights path {pretrained_weights} is not a file.")

        # If we are continuing training from a snapshot, we need to rename the old logs
        if starting_epoch > 0:
            os.rename(logs_path+"training_log.txt",
                      logs_path+f"training_log_{time.time()}.txt")

    # Initialize logs directory
    os.makedirs(logs_path+"training_data/", exist_ok=True)
    os.makedirs(logs_path+"snapshots/", exist_ok=True)

    with open(logs_path+"training_details.txt",'w') as ff:
            ff.write(f"-- BEGIN OF DETAILS --\n")
            ff.write(f"Model type: {model_type}\n")
            if hasattr(model, "arch"):
                ff.write(f" - architecture: {model.arch}\n")
                ff.write(f" - fft method: {model.fft_method}\n")
                ff.write(f" - resolutions: {model.resolutions}\n")
                ff.write(f" - spins: {model.spins}\n")
                ff.write(f" - widths: {model.widths}\n")
                ff.write(f" - nonlinearity: {model.nonlinearity_type}\n")
                ff.write(f" - use_batch_norm: {model.use_batch_norm}\n")
                ff.write(f" - use_layer_norm: {model.use_layer_norm}\n")
                ff.write(f" - recompute: {model.recompute}\n")
            ff.write(f"Experiment: {experiment}\n")
            if task.data == "mnist":
                ff.write(f" - type of stitching (train): {train_dataset.mode}\n")
                ff.write(f" - type of stitching (val): {val_dataset.mode}\n")
            ff.write(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
            ff.write(f"Total epochs: {total_epochs}\n")
            ff.write(f"Early stopping epochs: {early_stopping}\n")
            ff.write(f"Save every: {save_every}\n")
            ff.write(f"Optimizer: {optimizer}\n")
            ff.write(f"Batch size: {batch_size}\n")
            ff.write(f"FFT method (if applicable): {fft_method}\n")
            ff.write(f"Dataset size: {dataset_size}\n")
            ff.write(f"rotate: {rotate}\n")
            ff.write(f"Gradient accumulation #steps: {gradient_accumulation}\n")
            ff.write(f"Log id: {log_id}\n")
            ff.write("---------------------\n")
            ff.write(f"Full params dict: {params}\n")
            ff.write(f"-- END OF DETAILS --\n")

    with open(logs_path+"training_log.txt",'w') as ff:
            ff.write(f"-- BEGIN OF LOG --\n")

    torch.autograd.set_detect_anomaly(detect_anomaly)
    trainer = Trainer(gpu_id=gpu_id,
                      model = model,
                      in_datatype = task.in_datatype,
                      out_datatype = task.out_datatype,
                      train_dataloader = train_dataloader,
                      val_dataloader = val_dataloader,
                      optimizer = optimizer,
                      lossfnc = lossfnc,
                      save_every = save_every,
                      fft_method = fft_method,
                      logs_path = logs_path,
                      autoencoding = task.autoencoding,
                      rotate = rotate,
                      share_d_matrices = spec.share_d_matrices,
                      gradient_accumulation=gradient_accumulation,
                      verbose = verbose,
                      )

    trainer.train(max_epochs=total_epochs, early_stopping=early_stopping, starting_epoch=starting_epoch)

    with open(logs_path+"training_log.txt",'a') as ff:
            ff.write(f"-- END OF LOG --\n")

    # Rescaling the loss range for plotting
    max_loss = 1.000 if task.classification else 10
    plot_data(trainer, logs_path, max_loss=max_loss)
