import os
import pickle
import time

import torch

from training_utils import plot_data

from SphericalDataset import WeatherDataset, SphericalMNISTDataset
from utils.data_utils import make_dataloader

from modules.CNNUNet import CNNUNet
from modules.CNNAutoencoder import CNNAutoencoder512
from modules.MNIST_CNN import CNNMNISTClassifier


from modules.SO3UNet import SO3UNet
from modules.SO3Autoencoder import SO3Autoencoder
from modules.SO3CNN import SO3CNN
from modules.MNIST_SO3 import SO3MNISTClassifier

from modules.losses import S2MSELoss

from trainer_tester import Trainer

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
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 fft_method:str = "full", # Only for SO(3) models
                 dataset_size: int = None,
                 rotate: str = "no",
                 log_id: str = None,
                 preload_dataset:bool = True,
                 gradient_accumulation: int = 1,
                 verbose: bool = False,
                 ):
    """
    Run the training of the model with the given parameters.

    Parameters:
    - gpu_id: int, id of the GPU to use
    - params: dict, dictionary containing all parameters (if provided, overrides other parameters)
    - model_type: str, type of the model to use (e.g., "SO3UNet", "CNNUNet", etc.)
    - nonlinearity: str, type of nonlinearity to use
    - experiment: str, name of the experiment (e.g., "wind->wind", "temp->wind", etc.)
    - total_epochs: int, total number of epochs to train
    - starting_epoch: int, epoch number to start training from (useful for resuming)
    - pretrained_weights: str, path to pretrained weights to initialize the model
    - early_stopping: int, number of epochs to wait for improvement before stopping
    - save_every: int, how often to save the model
    - batch_size: int, size of the batches for training
    - use_batch_norm: bool, whether to use batch normalization
    - fft_method: str, method to use for FFT (e.g., "full", "delta", etc.)
    - dataset_size: int, size of the dataset to use (from 1 to 10)
    - rotate: str, rotation method to use (e.g., "no", "random", "beta", "full")
    - log_id: str, unique identifier for this run of this experiment (for logging purposes)
    - preload_dataset: bool, whether to preload the whole dataset into RAM memory
    - gradient_accumulation: int, optimiser steps are taken every this many batches
    - verbose: bool, whether to print detailed logs during training

    Experiments:
    - "wind->wind": Wind speed prediction the following day from wind speed
    - "temp->wind": Wind speed prediction the same day from temperature
    - "autoencoder": Autoencoder for wind speed
    - "MNIST-scalar-classification": MNIST classification with scalar input
    - "MNIST-vf-classification": MNIST classification with vector-field input
    - "MNIST-scalar-to-vector": MNIST scalar to vector prediction
    - "MNIST-vector-to-scalar": MNIST vector to scalar prediction

    Model types:
    - "SO3UNet": SO(3) equivariant UNet with full SO(3) layers
    - "SpinsSO3UNet": SO(3) equivariant UNet with spins layers
    - "CNNUNet": CNN UNet
    - "SO3autoencoder": SO(3) equivariant autoencoder with full SO(3) layers
    - "SpinsSO3autoencoder": SO(3) equivariant autoencoder with spins layers
    - "CNNautoencoder": CNN autoencoder
    - "SO3CNN": SO(3) equivariant flat CNN (no pooling/skip connections) for ablation
    - "SpinsSO3CNN": SO(3) equivariant flat CNN with spins layers for ablation
    - "SO3MNIST": SO(3) equivariant UNet/Classifier for MNIST with full SO(3) layers
    - "SpinsSO3MNIST": SO(3) equivariant UNet/Classifier for MNIST with spins layers
    - "CNNMNIST": CNN for MNIST classification
    """

    ALLOWED_EXPERIMENTS = ["wind->wind", "temp->wind", "autoencoder", "MNIST-scalar-classification",
                "MNIST-vf-classification", "MNIST-scalar-to-vector", "MNIST-vector-to-scalar"]
    ALLOWED_MODEL_TYPES = ["SO3UNet", "SpinsSO3UNet", "CNNUNet", "SO3autoencoder",
                "SpinsSO3autoencoder", "CNNautoencoder", "SO3CNN", "SpinsSO3CNN",
                "SO3MNIST", "SpinsSO3MNIST", "CNNMNIST",
                "SO3UNetOnlyOneSpin", "SpinsUNetOnlyOneSpin", "SO3MNISTOnlyOneSpin", "SpinsSO3MNISTOnlyOneSpin"] # Ablations
    ALLOWED_ROTATIONS = ["no", "offset", "beta", "full"]
    ALLOWED_FFT_METHODS = ["full",] #Delta is not currently supported after refactor
    ALLOWED_NONLINEARITIES = [None, "magnitude", "phase_collapse", "relu", "leaky_relu"]

    if params is not None:
        # Required: a schedule entry without one of these is a typo, not a default.
        experiment = params['experiment']
        model_type = params['model_type']
        total_epochs = params['total_epochs']
        batch_size = params['batch_size']
        rotate = params['rotate']

        fft_method = params.get('fft_method', 'full')
        early_stopping = params.get('early_stopping', total_epochs)
        starting_epoch = params.get('starting_epoch', 0)
        save_every = params.get('save_every', -1)  # Do not save by default
        pretrained_weights = params.get('pretrained_weights', None)
        use_batch_norm = params.get('use_batch_norm', False)
        use_layer_norm = params.get('use_layer_norm', False)
        log_id = params.get('log_id', None)
        dataset_size = params.get('dataset_size', None)  # None: not a weather experiment
        preload_dataset = params.get('preload_dataset', preload_dataset)
        verbose = params.get('verbose', verbose)
        gradient_accumulation = max(params.get('gradient_accumulation', 1), 1)

        if 'nonlinearity' in params:
            nonlinearity = params['nonlinearity']
        elif model_type in ["SO3UNet", "SO3autoencoder", "SO3CNN", "SO3MNIST"]:
            nonlinearity = "leaky_relu"
        elif model_type in ["SpinsSO3UNet", "SpinsSO3autoencoder", "SpinsSO3CNN",
                            "SpinsSO3MNIST"]:
            nonlinearity = 'magnitude'
        else:
            nonlinearity = None  # CNNs
    
    # Parameters check
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}.")
    if experiment not in ALLOWED_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}.")
    if rotate not in ALLOWED_ROTATIONS:
        raise ValueError(f"Unknown rotation type: {rotate}.")
    if fft_method is not None and fft_method not in ALLOWED_FFT_METHODS:
        raise ValueError(f"Unknown FFT method: {fft_method}.")
    if nonlinearity not in ALLOWED_NONLINEARITIES:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}.")
    if use_batch_norm and use_layer_norm:
        raise ValueError(f"Cannot use BOTH batch normalization and layer normalization.")

    if fft_method is None:
        fft_method = "full"

    if log_id is None:
        log_id = str(int(time.time())) #This is the id for the unique run
    else:
        log_id = str(log_id)
    
    DEVICE = torch.device("cuda", gpu_id)
    autoencoding = False # Default is False, set to True for autoencoder experiments
    share_d_matrices = False

    if rotate == "full":
        rotations_list = pickle.load(open("precomp/FULL_VALIDATION_ROTATIONS.list", "rb"))
    elif rotate == "beta":
        rotations_list = pickle.load(open("precomp/BETA_VALIDATION_ROTATIONS.list", "rb"))
    else:
        rotations_list = None
    
    ##############################
    ###      WIND -> WIND      ###
    ##############################
    if experiment == "wind->wind":
        in_datatype = "complex-vector"
        out_datatype = "complex-vector"

        train_folders = [str(year) for year in range(2000,2000+dataset_size)]
        train_dataset = WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
                                                    folders=train_folders,
                                       train=True,
                                       in_datatype="vector", out_datatype="vector")
        val_dataset = WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
                                     folders=["2020","2021"],
                                     train=False,
                                     in_datatype="vector", out_datatype="vector")
        train_dataset.preload()
        val_dataset.preload()

        if rotate != "no":
            val_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)

        train_dataloader = make_dataloader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
        val_dataloader = make_dataloader(val_dataset, batch_size=batch_size,
                                         shuffle=False)
        
        # model type
        if model_type == "SO3UNet":
            mode = "SO3"
            share_d_matrices = True
            model = SO3UNet(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 100, 50, 100, 179],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity= nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SO3UNetOnlyOneSpin":
            mode = "SO3"
            share_d_matrices = True
            model = SO3UNet(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 100, 50, 100, 179],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[1,], [1,], [1,], [1,], [1,], [1,]],
                            nonlinearity= nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SpinsSO3UNet":
            mode = "SPINS"
            model = SO3UNet(layer_style="spins",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 150, 100, 50, 100, 150, 179],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SpinsUNetOnlyOneSpin":
            mode = "SPINS"
            model = SO3UNet(layer_style="spins",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 150, 100, 50, 100, 150, 179],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [1,], [1,], [1,], [1,], [1,], [1,], [1,]],
                            nonlinearity = "magnitude",
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SO3CNN":
            mode = "SO3CNN"
            share_d_matrices = True
            model = SO3CNN(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolution = 179,
                            widths = [1,2,4,2,1],
                            spins = [[1,],[0,1],[0,1],[0,1],[1,]],
                            nonlinearity= nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SpinsSO3CNN":
            mode = "SPINS"
            model = SO3CNN(layer_style="spins",
                            n_alpha = 360, n_beta=179,
                            resolution = 179,
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "CNNUNet":
            mode = "CNN"
            model = CNNUNet(n_alpha=360, n_beta=179, in_datatype="complex-vector", out_datatype="complex-vector")
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        else:
            raise Exception(f"Combination {model_type} and {experiment} not supported.")
        
        lossfnc = S2MSELoss()
    
    ##############################
    ###      TEMP -> WIND      ###
    ##############################
    elif experiment == "temp->wind":
        in_datatype = "complex-scalar"
        out_datatype = "complex-vector"

        train_folders = [str(year) for year in range(2000,2000+dataset_size)]
        train_dataset = WeatherDataset(root = "dataset/temp->wind/weekly_coarse_torch/",
                                               folders=train_folders,
                                               train=True,
                                               in_datatype="scalar", out_datatype="vector")
        val_dataset = WeatherDataset(root = "dataset/temp->wind/weekly_coarse_torch/",
                                             folders=["2020","2021"],
                                             train=False,
                                             in_datatype="scalar", out_datatype="vector")
        train_dataset.preload()
        val_dataset.preload()

        if rotate != "no":
            val_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)

        train_dataloader = make_dataloader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
        val_dataloader = make_dataloader(val_dataset, batch_size=batch_size,
                                         shuffle=False)

        # model type
        if model_type == "SO3UNet":
            mode = "SO3"
            share_d_matrices = True
            model = SO3UNet(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 100, 50, 100, 179],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SpinsSO3UNet":
            mode = "SPINS"
            model = SO3UNet(layer_style="spins",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 150, 100, 50, 100, 150, 179],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "SO3CNN":
            mode = "SO3"
            share_d_matrices = True
            model = SO3CNN(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolution = 179,
                            widths = [1, 1],
                            spins = [[0,], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SpinsSO3CNN":
            mode = "SPINS"
            model = SO3CNN(layer_style="spins",
                            n_alpha = 360, n_beta=179,
                            resolution = 179,
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "CNNUNet":
            mode = "CNN"
            model = CNNUNet(n_alpha=360, n_beta=179, in_datatype="complex-scalar", out_datatype="complex-vector")
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        else:
            raise Exception(f"Combination {model_type} and {experiment} not supported.")

        lossfnc = S2MSELoss()
    
    #############################
    ###      AUTOENCODER      ###
    #############################
    elif experiment == "autoencoder":
        in_datatype = "complex-vector"
        out_datatype = "complex-vector"
        autoencoding = True
        train_folders = [str(year) for year in range(2000,2000+dataset_size)]
        train_dataset = WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
                                        folders=train_folders,
                                        train=True,
                                        in_datatype="vector", out_datatype="vector")
        val_dataset = WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
                                        folders=["2020","2021"],
                                        train=False,
                                        in_datatype="vector", out_datatype="vector")
        train_dataset.preload()
        val_dataset.preload()

        if rotate != "no":
            val_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)

        train_dataloader = make_dataloader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
        val_dataloader = make_dataloader(val_dataset, batch_size=batch_size,
                                         shuffle=False)
        
        # model type
        if model_type == "SO3autoencoder": 
            mode = "SO3"
            share_d_matrices = True
            model = SO3Autoencoder(layer_style = "so3",
                            n_alpha = 360, n_beta = 179,
                            resolutions = [179, 100, 50, 5, 50, 100, 179],
                            widths = [1, 2, 4, 7, 7, 4, 2, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "SpinsSO3autoencoder": 
            mode = "SPINS"
            model = SO3Autoencoder(layer_style = "spins",
                            n_alpha = 360, n_beta = 179,
                            resolutions = [179, 150, 100, 50, 5, 50, 100, 150, 179],
                            widths = [1, 16, 24, 32, 7, 7, 32, 24, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "CNNautoencoder":
            mode = "CNN"
            model = CNNAutoencoder512()
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        else:
            raise Exception(f"Combination {model_type} and {experiment} not supported.")
        
        lossfnc = S2MSELoss()

    #######################
    ###      MNIST      ###
    #######################

    # SCALAR MNIST CLASSIFICATION
    elif experiment == "MNIST-scalar-classification":
        in_datatype = "complex-scalar"
        out_datatype = "label"

        train_dataset = SphericalMNISTDataset(split = "train", mode = "spherical",
                                              n_alpha = 120, n_beta = 59,
                                              in_datatype = "scalar", out_datatype = "label",
                                              rotation_type = rotate, compute_device = DEVICE,
                                              caching = preload_dataset, verbose = verbose)

        val_dataset = SphericalMNISTDataset(split = "val", mode = "spherical",
                                            n_alpha = 120, n_beta = 59,
                                            in_datatype = "scalar", out_datatype = "label",
                                            rotation_type = rotate, rotations_list = rotations_list,
                                            compute_device = DEVICE,
                                            caching = preload_dataset, verbose = verbose)

        train_dataloader = make_dataloader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
        val_dataloader = make_dataloader(val_dataset, batch_size=batch_size,
                                         shuffle=False)
        lossfnc = torch.nn.CrossEntropyLoss()

        if model_type == "SO3MNIST": 
            mode = "SO3"
            model = SO3MNISTClassifier(n_alpha = 120, n_beta = 59, head = True,
                                       layer_style = "so3", fft_method = "full",
                                       widths = [1, 2, 8, 32, 128, 512],
                                       resolutions = [59, 38, 14, 8, 2, 0],
                                       spins = [[0,], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       nonlinearity = nonlinearity,
                                       use_batch_norm = use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "SO3MNISTOnlyOneSpin": 
            mode = "SO3"
            model = SO3MNISTClassifier(n_alpha = 120, n_beta = 59, head = True,
                                       layer_style = "so3", fft_method = "full",
                                       widths = [1, 2, 8, 32, 128, 512],
                                       resolutions = [59, 38, 14, 8, 2, 0],
                                       spins = [[0,], [0,], [0,], [0,], [0,], [0,]],
                                       nonlinearity = nonlinearity,
                                       use_batch_norm = use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "SpinsSO3MNIST": 
            mode = "SPINS"
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59, head=True,
                                       layer_style="spins", fft_method="full",
                                       widths=[1, 4, 16, 32, 64, 128, 256, 512],
                                       resolutions=[59, 50, 40, 30, 20, 10, 2, 0],
                                       spins=[[0,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       nonlinearity=nonlinearity,
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "SpinsSO3MNISTOnlyOneSpin": 
            mode = "SPINS"
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59, head=True,
                                       layer_style="spins", fft_method="full",
                                       widths=[1, 4, 16, 32, 64, 128, 256, 512],
                                       resolutions=[59, 50, 40, 30, 20, 10, 2, 0],
                                       spins=[[0,], [0,], [0,], [0,], [0,], [0,], [0,], [0,]],
                                       nonlinearity="magnitude",
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "CNNMNIST":
            mode = "CNN"
            model = CNNMNISTClassifier(in_datatype="complex-scalar",
                                       n_alpha=120, n_beta=59,
                                       widths=[1, 8, 32, 128, 256, 512])
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        else:
            raise Exception(f"Combination {model_type} and {experiment} not supported.")
        
    # VECTOR-FIELD MNIST CLASSIFICATION
    elif experiment == "MNIST-vf-classification":
        in_datatype = "complex-vector"
        out_datatype = "label"

        train_dataset = SphericalMNISTDataset(split="train", mode="spherical",
                                              n_alpha=120, n_beta=59,
                                              in_datatype="vector", out_datatype="label",
                                              rotation_type=rotate, compute_device=DEVICE,
                                              caching=preload_dataset, verbose=verbose)
        
        val_dataset = SphericalMNISTDataset(split="val", mode="spherical",
                                            n_alpha=120, n_beta=59,
                                            in_datatype="vector", out_datatype="label",
                                            rotation_type=rotate, rotations_list=rotations_list,
                                            compute_device=DEVICE,
                                            caching=preload_dataset, verbose=verbose)

        train_dataloader = make_dataloader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
        val_dataloader = make_dataloader(val_dataset, batch_size=batch_size,
                                         shuffle=False)
        
        lossfnc = torch.nn.CrossEntropyLoss()
        
        if model_type == "SO3MNIST": 
            mode = "SO3"
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59,
                                       layer_style="so3", fft_method="full",
                                       #widths=[1, 4, 16, 64, 256, 512],
                                       #resolutions=[59, 50, 25, 10, 2, 0],
                                       widths = [1, 2, 8, 32, 128, 512],
                                       resolutions = [59, 38, 14, 8, 2, 0],
                                       spins=[[1,], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       nonlinearity=nonlinearity,
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "SpinsSO3MNIST":
            mode = "SPINS"
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59,
                                       layer_style="spins", fft_method="full",
                                       widths=[1, 4, 16, 64, 256, 512],
                                       resolutions=[59, 50, 25, 10, 2, 0],
                                       spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       nonlinearity=nonlinearity,
                                       use_batch_norm = use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       )
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        elif model_type == "CNNMNIST":
            mode = "CNN"
            model = CNNMNISTClassifier(in_datatype="complex-vector",
                                       n_alpha=120, n_beta=59,
                                       widths=[2, 8, 32, 128, 256, 512])
            optimizer = torch.optim.Adam(model.parameters(),1e-3)
        else:
            raise Exception(f"Combination {model_type} and {experiment} not supported.")

    # SCALAR-TO-VECTOR MNIST
    elif experiment == "MNIST-scalar-to-vector":
        in_datatype = "complex-scalar"
        out_datatype = "complex-vector"

        train_dataset = SphericalMNISTDataset(split="train", mode="spherical",
                                              n_alpha=120, n_beta=59,
                                              in_datatype="scalar", out_datatype="vector",
                                              rotation_type=rotate, compute_device=DEVICE,
                                              caching=preload_dataset, verbose=verbose)
        
        val_dataset = SphericalMNISTDataset(split="val", mode="spherical",
                                            n_alpha=120, n_beta=59,
                                            in_datatype="scalar", out_datatype="vector",
                                            rotation_type=rotate, rotations_list=rotations_list,
                                            compute_device=DEVICE,
                                            caching=preload_dataset, verbose=verbose)

            
        train_dataloader = make_dataloader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
        val_dataloader = make_dataloader(val_dataset, batch_size=batch_size,
                                         shuffle=False)

        lossfnc = S2MSELoss(n_beta=59)

        if model_type == "SO3UNet":
            mode = "SO3"
            model = SO3UNet(layer_style="so3", fft_method="full",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 25, 50, 59],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity=nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SpinsSO3UNet":
            mode = "SPINS"
            model = SO3UNet(layer_style="spins",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 40, 30, 40, 50, 59],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity=nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "CNNUNet":
            mode = "CNN"
            model = CNNUNet(n_alpha=120, n_beta=59, in_datatype="complex-scalar", out_datatype="complex-vector")
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        else:
            raise Exception(f"Combination {model_type} and {experiment} not supported.")

    # VECTOR-TO-SCALAR MNIST
    elif experiment == "MNIST-vector-to-scalar":
        in_datatype = "complex-vector"
        out_datatype = "complex-scalar"

        train_dataset = SphericalMNISTDataset(split="train", mode="spherical",
                                              n_alpha=120, n_beta=59,
                                              in_datatype="vector", out_datatype="scalar",
                                              rotation_type=rotate, compute_device=DEVICE,
                                              caching=preload_dataset, verbose=verbose)
        
        val_dataset = SphericalMNISTDataset(split="val", mode="spherical",
                                            n_alpha=120, n_beta=59,
                                            in_datatype="vector", out_datatype="scalar",
                                            rotation_type=rotate, rotations_list=rotations_list,
                                            compute_device=DEVICE,
                                            caching=preload_dataset, verbose=verbose)


        train_dataloader = make_dataloader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
        val_dataloader = make_dataloader(val_dataset, batch_size=batch_size,
                                         shuffle=False)

        lossfnc = S2MSELoss(n_beta=59)

        if model_type == "SO3UNet":
            mode = "SO3"
            model = SO3UNet(layer_style="so3", fft_method="full",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 25, 50, 59],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,]],
                            nonlinearity=nonlinearity,
                            use_batch_norm=use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "SpinsSO3UNet":
            mode = "SPINS"
            model = SO3UNet(layer_style="spins",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 40, 50, 40, 50, 59],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,]],
                            nonlinearity=nonlinearity,
                            use_batch_norm=use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            )
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        elif model_type == "CNNUNet":
            mode = "CNN"
            model = CNNUNet(n_alpha=120, n_beta=59, in_datatype="complex-vector", out_datatype="complex-scalar")
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
        else:
            raise Exception(f"Combination {model_type} and {experiment} not supported.")

    ### Sanity check ###
    else:
        raise Exception(f"Experiment {experiment} not defined.")

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
            old_logs_path = get_experiment_path(experiment, mode, rotate, dataset_size, log_id)+"training_log.txt"
            os.rename(old_logs_path,
                      get_experiment_path(experiment, mode, rotate, dataset_size, log_id)+
                      f"training_log_{time.time()}.txt")

    # Initialize logs directory
    for epoch in range(total_epochs):
        os.makedirs(os.path.dirname(
            get_experiment_path(experiment, mode, rotate, dataset_size, log_id)\
                                    +"training_data/"), exist_ok=True)
        os.makedirs(os.path.dirname(
            get_experiment_path(experiment, mode, rotate, dataset_size, log_id)\
                                    +"snapshots/"), exist_ok=True)

    with open(get_experiment_path(experiment, mode, rotate, dataset_size, log_id)\
              +"training_details.txt",'w') as ff:
            ff.write(f"-- BEGIN OF DETAILS --\n")
            ff.write(f"Model type: {model_type}\n")
            if mode == "SO3" or mode == "SPINS":
                ff.write(f" - layer style: {model.layer_style}\n")
                ff.write(f" - fft method: {model.fft_method}\n")
                ff.write(f" - resolutions: {model.resolutions}\n")
                ff.write(f" - spins: {model.spins}\n")
                ff.write(f" - widths: {model.widths}\n")
                ff.write(f" - nonlinearity: {model.nonlinearity_type}\n")
                ff.write(f" - use_batch_norm: {model.use_batch_norm}\n")
                ff.write(f" - use_layer_norm: {model.use_layer_norm}\n")
            ff.write(f"Experiment: {experiment}\n")
            if experiment == "MNIST-scalar-classification" or experiment == "MNIST-vf-classification" or experiment == "MNIST-scalar-to-vector" or experiment == "MNIST-vector-to-scalar":
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

    with open(get_experiment_path(experiment, mode, rotate, dataset_size, log_id)\
              +"training_log.txt",'w') as ff:
            ff.write(f"-- BEGIN OF LOG --\n")
            
    torch.autograd.set_detect_anomaly(True)
    trainer = Trainer(gpu_id=gpu_id,
                      model = model,
                      in_datatype = in_datatype,
                      out_datatype = out_datatype,
                      train_dataloader = train_dataloader,
                      val_dataloader = val_dataloader,
                      optimizer = optimizer,
                      lossfnc = lossfnc,
                      save_every = save_every,
                      fft_method = fft_method,
                      logs_path = get_experiment_path(experiment, mode, rotate, dataset_size, log_id),
                      autoencoding = autoencoding,
                      rotate = rotate,
                      share_d_matrices = share_d_matrices,
                      gradient_accumulation=gradient_accumulation,
                      verbose = verbose,
                      )
    
    
    trainer.train(max_epochs=total_epochs, early_stopping=early_stopping, starting_epoch=starting_epoch)

    with open(get_experiment_path(experiment, mode, rotate, dataset_size, log_id)\
              +"training_log.txt",'a') as ff:
            ff.write(f"-- END OF LOG --\n")
    
    # Rescaling the loss range for plotting
    if experiment == "MNIST-scalar-classification" or experiment == "MNIST-vf-classification":
        max_loss = 1.000
    else:
        max_loss = 10

    plot_data(trainer, get_experiment_path(experiment, mode, rotate, dataset_size, log_id),
              max_loss=max_loss)