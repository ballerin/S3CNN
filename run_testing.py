import os
import pickle

import torch

from SphericalDataset import WeatherDataset, SphericalMNISTDataset
from utils.data_utils import make_dataloader

from modules.CNNUNet import CNNUNet
from modules.CNNAutoencoder import CNNAutoencoder512
from modules.MNIST_CNN import CNNMNISTClassifier


from modules.SO3UNet import SO3UNet
from modules.SO3CNN import SO3CNN
from modules.SO3Autoencoder import SO3Autoencoder
from modules.MNIST_SO3 import SO3MNISTClassifier

from modules.losses import S2RMSELoss

from trainer_tester import Trainer, Tester

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
        experiment = params['experiment']
        model_type = params['model_type']
        batch_size = params['batch_size']
        rotate = params['rotate']
        path = params['path']

        fft_method = params.get('fft_method', 'full')
        if 'nonlinearity' not in params.keys():
            if model_type in ["SO3UNet", "SO3autoencoder", "SO3MNIST"]:
                nonlinearity = "leaky_relu"
            elif model_type in ["SpinsSO3UNet", "SpinsSO3autoencoder", "SpinsSO3MNIST"]:
                nonlinearity = "magnitude"
            else:
                #CNNs
                nonlinearity = None
        else:
            nonlinearity = params['nonlinearity']
        if 'use_batch_norm' not in params.keys():
            use_batch_norm = False
        else:
            use_batch_norm = params['use_batch_norm']
        if 'use_layer_norm' not in params.keys():
            use_layer_norm = False
        else:
            use_layer_norm = params['use_layer_norm']
            
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
    
    DEVICE = torch.device("cuda", gpu_id)
    autoencoding = False
    
    weights_path = os.path.join(path, "snapshots/checkpoint_best.pt")

    if rotate == "no":
        log_path = os.path.join(path, "NR_testing_log.txt")
        losses_path = os.path.join(path, "NR_test_losses.pckl")
        accuracies_path = os.path.join(path, "NR_test_accuracies.pckl")
    elif rotate == "beta":
        log_path = os.path.join(path, "RB_testing_log.txt")
        losses_path = os.path.join(path, "RB_test_losses.pckl")
        accuracies_path = os.path.join(path, "RB_test_accuracies.pckl")
    elif rotate == "full":
        log_path = os.path.join(path, "RF_testing_log.txt")
        losses_path = os.path.join(path, "RF_test_losses.pckl")
        accuracies_path = os.path.join(path, "RF_test_accuracies.pckl")

    if rotate == "full":
        rotations_list = pickle.load(open("precomp/FULL_TEST_ROTATIONS.list", "rb"))
    elif rotate == "beta":
        rotations_list = pickle.load(open("precomp/BETA_TEST_ROTATIONS.list", "rb"))
    else:
        rotations_list = None
    
    #Initialization
    train_dataloader = None

    ##############################
    ###      WIND -> WIND      ###
    ##############################
    if experiment == "wind->wind":
        in_datatype = "complex-vector"
        out_datatype = "complex-vector"

        test_dataset = WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
                                      folders = ["2022","2023"],
                                      train = False,
                                      in_datatype = "vector", out_datatype = "vector")
        test_dataset.preload()

        test_dataloader = make_dataloader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
        
        if rotate != "no":
            test_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)
        
        lossfnc = S2RMSELoss(n_beta=179)

        if model_type == "SO3UNet":
             model = SO3UNet(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 100, 50, 100, 179],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity=nonlinearity,
                            use_batch_norm=use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
             model.load_state_dict(
                  torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "SO3UNetOnlyOneSpin":
            model = SO3UNet(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 100, 50, 100, 179],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[1,], [1,], [1,], [1,], [1,], [1,]],
                            nonlinearity= "magnitude", #Forcing magnitude nonlinearity, as spins are all 1
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "SpinsSO3UNet":
            model = SO3UNet(layer_style = "spins",
                            n_alpha = 360, n_beta = 179,
                            resolutions = [179, 150, 100, 50, 100, 150, 179],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "SpinsUNetOnlyOneSpin":
            model = SO3UNet(layer_style="spins",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 150, 100, 50, 100, 150, 179],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [1,], [1,], [1,], [1,], [1,], [1,], [1,]],
                            nonlinearity = "magnitude",
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "SO3CNN":
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
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "SpinsSO3CNN":
            model = SO3CNN(layer_style="spins",
                            n_alpha = 360, n_beta=179,
                            resolution = 179,
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "CNNUNet":
            model = CNNUNet(n_alpha=360, n_beta=179, in_datatype="complex-vector", out_datatype="complex-vector").to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
    
    ##############################
    ###      TEMP -> WIND      ###
    ##############################
    elif experiment == "temp->wind":
        in_datatype = "complex-scalar"
        out_datatype = "complex-vector"

        test_dataset = WeatherDataset(root = "dataset/temp->wind/weekly_coarse_torch/",
                                      folders = ["2022","2023"],
                                      train = False,
                                      in_datatype="scalar", out_datatype="vector")
        test_dataset.preload()

        lossfnc = S2RMSELoss(n_beta=179)

        if rotate != "no":
            test_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)
        
        test_dataloader = make_dataloader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
        
        
        if model_type == "SO3UNet":
            model = SO3UNet(layer_style="so3",
                            fft_method="full",
                            n_alpha = 360, n_beta=179,
                            resolutions = [179, 100, 50, 100, 179],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                  torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "SpinsSO3UNet":
            model = SO3UNet(layer_style = "spins",
                            n_alpha = 360, n_beta = 179,
                            resolutions = [179, 150, 100, 50, 100, 150, 179],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity=nonlinearity,
                            use_batch_norm=use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "CNNUNet":
            model = CNNUNet(n_alpha=360, n_beta=179, in_datatype="complex-scalar", out_datatype="complex-vector").to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        
        

    #############################
    ###      AUTOENCODER      ###
    #############################
    elif experiment == "autoencoder":
        autoencoding = True
        in_datatype = "complex-vector"
        out_datatype = "complex-vector"

        test_dataset = WeatherDataset(root = "dataset/wind->wind/weekly_coarse_torch/",
                                      folders = ["2022","2023"],
                                      train = False,
                                      in_datatype = "vector", out_datatype = "vector")
        test_dataset.preload()

        
        if rotate != "no":
            test_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)
        
        test_dataloader = make_dataloader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
        
        lossfnc = S2RMSELoss(n_beta=179)

        if model_type == "SO3autoencoder":
            model = SO3Autoencoder(layer_style = "so3",
                            n_alpha = 360, n_beta = 179,
                            resolutions = [179, 100, 50, 5, 50, 100, 179],
                            widths = [1, 2, 4, 7, 7, 4, 2, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "SpinsSO3autoencoder":
            model = SO3Autoencoder(layer_style = "spins",
                            n_alpha = 360, n_beta = 179,
                            resolutions = [179, 150, 100, 50, 5, 50, 100, 150, 179],
                            widths = [1, 16, 24, 32, 7, 7, 32, 24, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        elif model_type == "CNNautoencoder":
            model = CNNAutoencoder512().to(DEVICE)
            model.load_state_dict(
                 torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
        
    #######################
    ###      MNIST      ###
    #######################

    # SCALAR MNIST CLASSIFICATION
    elif experiment == "MNIST-scalar-classification":
        in_datatype = "complex-scalar"
        out_datatype = "label"

        test_dataset = SphericalMNISTDataset(split="test", mode="spherical",
                                            n_alpha=120, n_beta=59,
                                            in_datatype="scalar", out_datatype="label",
                                            rotation_type=rotate, rotations_list=rotations_list,
                                            compute_device=DEVICE,
                                            caching=True, verbose=False)
        caching = True
        if not caching and rotate != "no":
            test_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)
        
        test_dataloader = make_dataloader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
        
        lossfnc = None  # No loss function needed for classification test, as we use accuracy

        if model_type == "SO3MNIST": 
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59, head=True,
                                       layer_style="so3", fft_method="full",
                                       widths = [1, 2, 8, 32, 128, 512],
                                       resolutions = [59, 38, 14, 8, 2, 0],
                                       spins=[[0,], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       nonlinearity=nonlinearity).to(DEVICE)
        elif model_type == "SO3MNISTOnlyOneSpin": 
            model = SO3MNISTClassifier(n_alpha = 120, n_beta = 59, head = True,
                                       layer_style = "so3", fft_method = "full",
                                       widths = [1, 2, 8, 32, 128, 512],
                                       resolutions = [59, 38, 14, 8, 2, 0],
                                       spins = [[0,], [0,], [0,], [0,], [0,], [0,]],
                                       nonlinearity = nonlinearity,
                                       use_batch_norm = use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       ).to(DEVICE)
        elif model_type == "SpinsSO3MNIST":
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59,
                                       layer_style="spins", fft_method="full",
                                       widths=[1, 4, 16, 32, 64, 128, 256, 512],
                                       resolutions=[59, 50, 40, 30, 20, 10, 2, 0],
                                       spins=[[0,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       nonlinearity=nonlinearity).to(DEVICE)
        elif model_type == "SpinsSO3MNISTOnlyOneSpin": 
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59, head=True,
                                       layer_style="spins", fft_method="full",
                                       widths=[1, 4, 16, 32, 64, 128, 256, 512],
                                       resolutions=[59, 50, 40, 30, 20, 10, 2, 0],
                                       spins=[[0,], [0,], [0,], [0,], [0,], [0,], [0,], [0,]],
                                       nonlinearity="magnitude",
                                       use_batch_norm=use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       ).to(DEVICE)
        elif model_type == "CNNMNIST":
            model = CNNMNISTClassifier(in_datatype="complex-scalar",
                                       n_alpha=120, n_beta=59,
                                       widths=[1, 8, 32, 128, 256, 512]).to(DEVICE)

        model.load_state_dict(
                torch.load(weights_path, map_location='cuda:'+str(gpu_id)), strict=True)
    
    # VECTOR-FIELD MNIST CLASSIFICATION
    elif experiment == "MNIST-vf-classification":
        in_datatype = "complex-vector"
        out_datatype = "label"

        test_dataset = SphericalMNISTDataset(split="test", mode="spherical",
                                            n_alpha=120, n_beta=59,
                                            in_datatype="vector", out_datatype="label")

        if rotate != "no":
            test_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)

        test_dataloader = make_dataloader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
        
        lossfnc = None  # No loss function needed for classification test, as we use accuracy
        
        if model_type == "SO3MNIST": 
            model = SO3MNISTClassifier(n_alpha = 120, n_beta = 59,
                                       layer_style = "so3", fft_method = "full",
                                       widths = [1, 2, 8, 32, 128, 512],
                                       resolutions = [59, 38, 14, 8, 2, 0],
                                       spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       nonlinearity = nonlinearity,
                                       use_batch_norm = use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       ).to(DEVICE)
        elif model_type == "SpinsSO3MNIST":
            model = SO3MNISTClassifier(n_alpha=120, n_beta=59,
                                       layer_style="spins", fft_method="full",
                                       widths=[1, 4, 16, 64, 256, 512],
                                       resolutions=[59, 50, 25, 10, 2, 0],
                                       spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,]],
                                       nonlinearity=nonlinearity,
                                       use_batch_norm = use_batch_norm,
                                       use_layer_norm = use_layer_norm,
                                       ).to(DEVICE)
        elif model_type == "CNNMNIST":
            model = CNNMNISTClassifier(in_datatype="complex-vector",
                                       n_alpha=120, n_beta=59,
                                       widths=[2, 8, 32, 128, 256, 512]).to(DEVICE)

        model.load_state_dict(
                torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
    
    # SCALAR-TO-VECTOR MNIST
    elif experiment == "mnist-scalar-to-vector":
        in_datatype = "complex-scalar"
        out_datatype = "complex-vector"

        test_dataset = SphericalMNISTDataset(split="test", mode="spherical",
                                             n_alpha=120, n_beta=59,
                                             in_datatype="scalar", out_datatype="vector")

        if rotate != "no":
            test_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)
            
        test_dataloader = make_dataloader(test_dataset, batch_size=batch_size,
                                          shuffle=True)

        lossfnc = S2RMSELoss(n_beta=59)

        if model_type == "SO3UNet":
            model = SO3UNet(layer_style="so3", fft_method="full",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 25, 50, 59],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
        elif model_type == "SpinsSO3UNet":
            model = SO3UNet(layer_style="spins",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 100, 50, 100, 50, 59],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[0,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [1,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
        elif model_type == "CNNUNet":
            model = CNNUNet(in_datatype="complex-scalar", out_datatype="complex-vector")

        model.load_state_dict(
                torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
    
    # VECTOR-TO-SCALAR MNIST
    elif experiment == "mnist-vector-to-scalar":
        in_datatype = "complex-vector"
        out_datatype = "complex-scalar"

        test_dataset = SphericalMNISTDataset(split="test", mode="spherical",
                                             n_alpha=120, n_beta=59,
                                             in_datatype="vector", out_datatype="scalar")

        if rotate != "no":
            test_dataset.rotate_from_list(device=DEVICE, rotations_list=rotations_list,
                                         rotation_type=rotate)
            
        test_dataloader = make_dataloader(test_dataset, batch_size=batch_size,
                                          shuffle=True)

        lossfnc = S2RMSELoss(n_beta=59)

        if model_type == "SO3UNet":
            model = SO3UNet(layer_style="so3", fft_method="full",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 25, 50, 59],
                            widths = [1, 2, 8, 30, 8, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
        elif model_type == "SpinsSO3UNet":
            model = SO3UNet(layer_style="spins",
                            n_alpha = 120, n_beta = 59,
                            resolutions = [59, 50, 100, 50, 100, 50, 59],
                            widths = [1, 4, 32, 128, 256, 48, 16, 1],
                            spins = [[1,], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,]],
                            nonlinearity = nonlinearity,
                            use_batch_norm = use_batch_norm,
                            use_layer_norm = use_layer_norm,
                            ).to(DEVICE)
        elif model_type == "CNNUNet":
            model = CNNUNet(in_datatype="complex-vector", out_datatype="complex-scalar")

        model.load_state_dict(
                torch.load(weights_path, weights_only=True, map_location='cuda:'+str(gpu_id)))
    
    with open(log_path,'w') as ff:
            ff.write(f"-- BEGIN OF LOG --\n")
    
    
    tester = Tester(model = model,
                    in_datatype = in_datatype,
                    out_datatype = out_datatype,
                    dataloader = test_dataloader,
                    lossfnc = lossfnc,
                    gpu_id = gpu_id,
                    fft_method = fft_method,
                    log_path = log_path,
                    losses_path = losses_path,
                    accuracies_path = accuracies_path,
                    autoencoding = autoencoding,
                    rotate = rotate,
                    train_dataloader = train_dataloader
                    )

    tester.test()

    with open(log_path,'a') as ff:
            ff.write(f"-- END OF LOG --\n")