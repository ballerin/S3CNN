import os
import torch
import random
from tqdm import tqdm, trange
import numpy as np
import pickle
from torch.utils.data import Dataset, random_split
from torchvision import transforms, utils, datasets
import gc

from modules.SO3Rotator import SO3Rotator
from modules.SO3lib import stitch_mnist_to_sphere_simple, stitch_mnist_to_sphere_proj,\
                        stitch_mnist_to_sphere_spherical, compute_gradients

#######################################
### SphericalDataset Template Class ###
#######################################
class SphericalDataset(Dataset):
    """
    Template for spherical datasets, such as MNIST on the sphere or ERA5.
    """

    def __init__(self, n_alpha:int, n_beta:int, in_datatype:str, out_datatype:str, random_seed:int=38059, verbose=False):
        """
        """
        super().__init__()

        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)

        self.in_datatype = in_datatype
        self.out_datatype = out_datatype

        self.random_seed = int(random_seed)
        
        self.is_preloaded = False
        self.preloaded_dataset = []

        self.verbose = verbose

        if self.n_alpha <= 0 or self.n_beta <= 0:
            raise Exception("n_alpha and n_beta must be positive integers.")
    
    def preload(self):
        """
        Preload the dataset into memory for faster access. In practice, this is feasible for small
        datasets like MNIST, and small sphere resolutions.
        This method should be called only once, and it will not be called again.
        """
        if self.is_preloaded:
            return
        for i in trange(len(self), disable=not self.verbose, desc="Preloading dataset"):
            x, y = self.__getitem__(i)
            self.preloaded_dataset.append((x, y))
            
            # Trigger garbage collection periodically to prevent memory buildup
            if i % 1000 == 0:
                gc.collect()
        self.is_preloaded = True
        # Final cleanup after preloading
        gc.collect()

    def __len__(self):
        if self.is_preloaded:
            return len(self.preloaded_dataset)
        else:
            return len(self.dataset)
    
    def _init_rotators(self, device):
        if self.in_datatype == "vector" and self.out_datatype == "scalar":
            rotatorX = SO3Rotator(device=device, n=1, n_alpha=self.n_alpha, n_beta=self.n_beta, 
                                  FFT_L=self.n_beta)
            rotatorY = SO3Rotator(device=device, n=0, n_alpha=self.n_alpha, n_beta=self.n_beta, 
                                  FFT_L=self.n_beta)
        elif self.in_datatype == "scalar" and self.out_datatype == "vector":
            rotatorX = SO3Rotator(device=device, n=0, n_alpha=self.n_alpha, n_beta=self.n_beta, 
                                  FFT_L=self.n_beta)
            rotatorY = SO3Rotator(device=device, n=1, n_alpha=self.n_alpha, n_beta=self.n_beta, 
                                  FFT_L=self.n_beta)
        elif self.in_datatype == "scalar" and self.out_datatype == "label":
            rotatorX = SO3Rotator(device=device, n=0, n_alpha=self.n_alpha, n_beta=self.n_beta, 
                                  FFT_L=self.n_beta)
            rotatorY = None
        elif self.in_datatype == "vector" and self.out_datatype == "label":
            rotatorX = SO3Rotator(device=device, n=1, n_alpha=self.n_alpha, n_beta=self.n_beta, 
                                  FFT_L=self.n_beta)
            rotatorY = None
        elif self.in_datatype == "vector" and self.out_datatype == "vector":
            rotatorX = SO3Rotator(device=device, n=1, n_alpha=self.n_alpha, n_beta=self.n_beta, 
                                  FFT_L=self.n_beta)
            rotatorY = rotatorX
        else:
            raise Exception(f"X={self.in_datatype} and Y={self.out_datatype} not implemented")
        return rotatorX, rotatorY
    
    def apply_offset(self, device:torch.device=torch.device("cpu"), offset:tuple[int]=(0,0,0)):
        """
        Apply an offset to the dataset. This is useful for datasets where the samples are projected
        on an hemisphere that makes sense to shift the samples by a certain offset.
        """
        if not self.is_preloaded:
            self.preload()
        
        rotatorX, rotatorY = self._init_rotators(device=device)
        try:
            for i in range(len(self)):
                X, Y = self.preloaded_dataset[i]
                if rotatorX is not None:
                    X = rotatorX(X[None,:].to(device), rotation=offset).detach()[0].cpu()
                if rotatorY is not None:
                    Y = rotatorY(Y[None,:].to(device), rotation=offset).detach()[0].cpu()
                self.preloaded_dataset[i] = [X, Y]
        finally:
            # Ensure cleanup happens even if an error occurs
            del rotatorX, rotatorY
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def randomly_rotate(self, device = torch.device('cpu')):
        """
        Rotate all samples in the dataset by a random rotation Y(beta).
        Do so using the specified device.
        """
        if not self.is_preloaded:
            self.preload()

        rotatorX, rotatorY = self._init_rotators(device = device)
        
        try:
            for i in range(len(self)):
                beta = np.random.randint(0,360)
                X, Y = self.preloaded_dataset[i]
                if rotatorX is not None:
                    X = rotatorX(X[None,:].to(device), beta=beta).detach()[0].cpu()
                if rotatorY is not None:
                    Y = rotatorY(Y[None,:].to(device), beta=beta).detach()[0].cpu()
                self.preloaded_dataset[i] = [X, Y]
                
                # Clear cache periodically to prevent accumulation
                if i % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
        finally:
            del rotatorX, rotatorY
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    def sequentially_rotate(self, device = torch.device('cpu')):
        """
        Rotate all samples in the dataset by a random rotation Y(beta).
        Do so using the specified device.
        """
        if not self.is_preloaded:
            self.preload()

        rotatorX, rotatorY = self._init_rotators(device = device)

        try:
            for i in range(len(self)):
                beta = i*(360//len(self))
                X, Y = self.preloaded_dataset[i]
                if rotatorX is not None:
                    X = rotatorX(X[None,:].to(device), beta=beta).detach()[0].cpu()
                if rotatorY is not None:
                    Y = rotatorY(Y[None,:].to(device), beta=beta).detach()[0].cpu()
                self.preloaded_dataset[i] = [X, Y]
                
                # Clear cache periodically to prevent accumulation
                if i % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
        finally:
            del rotatorX, rotatorY
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    def rotate_from_list(self, device:torch.device = torch.device('cpu'),
                         rotations_list:list = None, rotation_type:str = "full"):
        """
        Rotate all samples in the dataset by betas in a specified list.
        Do so using the specified device.
        """
        if rotations_list is None:
            raise Exception("Rotation list not provided.")
        elif (not (isinstance(rotations_list[0], list) or isinstance(rotations_list[0],tuple))) and rotation_type != "beta":
            raise Exception("Cannot perform full rotation without a ZYZ angles list")

        if not self.is_preloaded:
            self.preload()

        rotatorX, rotatorY = self._init_rotators(device = device)

        try:
            for i in range(len(self)):
                # Repeat the rotations if the list is shorter than the dataset
                
                if rotation_type == "beta":
                    beta = rotations_list[i%len(rotations_list)]
                elif rotation_type == "full":
                    rotation = rotations_list[i%len(rotations_list)]
                else:
                    raise Exception(f"Unknown rotation type: {rotation_type}")

                X, Y = self.preloaded_dataset[i]
                if rotatorX is not None:
                    if rotation_type == "full":
                        X = rotatorX(X[None,:].to(device), rotation=rotation).detach()[0].cpu()
                    elif rotation_type == "beta":
                        X = rotatorX(X[None,:].to(device), beta=beta).detach()[0].cpu()
                    else:
                        raise Exception(f"Unknown rotation type: {rotation_type}")
                if rotatorY is not None:
                    if rotation_type == "full":
                        Y = rotatorY(Y[None,:].to(device), rotation=rotation).detach()[0].cpu()
                    elif rotation_type == "beta":
                        Y = rotatorY(Y[None,:].to(device), beta=beta).detach()[0].cpu()
                    else:
                        raise Exception(f"Unknown rotation type: {rotation_type}")
                self.preloaded_dataset[i] = [X, Y]
                
                # Clear cache periodically to prevent accumulation
                if i % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
        finally:
            del rotatorX, rotatorY
            if device.type == 'cuda':
                torch.cuda.empty_cache()


#######################
### Spherical MNIST ###
#######################
class SphericalMNISTDataset(SphericalDataset):
    """
    Spherical MNIST dataset for training and testing spherical neural networks. 
    The dataset is based on the MNIST dataset, stitching the images onto a sphere.

    The encoding for vector fields is -v+iu, where u and v are the gradients of the MNIST image, x
    and y respectively.
    """

    def __init__(self, split="train",
                 n_alpha=120, n_beta=59,
                 mode="simple", caching=False,
                 in_datatype="scalar", out_datatype="label",
                 rotation_type = "no", rotations_list = None,
                 compute_device = torch.device("cpu"),
                 offset:tuple[int]=(0,0,0), verbose=False):
        """
        Parameters
        - train: bool, variable reflected to the original MNIST dataset.
        - mode: simple/equirectangular.
        - caching: bool, whether to cache the dataset in memory and retrieve it from there.
        - rotation_type: no/beta/full:
            - no: no rotation applied.
            - beta: rotate by a random angle beta around the y axis.
            - full: rotate by a random rotation (alpha, beta, gamma).
        - rotations_list: list of rotations to apply to the dataset. rotation_type cannot be "no".
        - in_datatype: scalar/vector.
        - out_datatype: label/vector/scalar.
        - offset: tuple of 3 ints, offset to apply to the stitched MNIST images on the sphere.
        - compute_device: device to use for computations that can be offloaded to GPU.
        - verbose: bool, whether to print progress bars.
        """
        super().__init__(n_alpha=n_alpha, n_beta=n_beta,
                         in_datatype=in_datatype, out_datatype=out_datatype,
                         verbose=verbose)

        self.split = split
        self.MNIST_train = (split == "train") or (split == "trainval")
        self.mode = mode
        self.offset = offset
        self.rotation_type = rotation_type

        ALLOWED_SPLITS = ["train", "test", "val", "trainval"]
        ALLOWED_MODES = ["simple", "equirectangular", "spherical"]
        ALLOWED_DATATYPES = ["scalar", "vector", "label"]
        ALLOWED_ROTATIONS = ["no", "beta", "full"]

        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Unknown split: {split}. Must be {ALLOWED_SPLITS}")
        if self.mode not in ALLOWED_MODES:
            raise ValueError(f"Unknown projection mode: {self.mode}. Must be {ALLOWED_MODES}")
        if self.in_datatype not in ALLOWED_DATATYPES:
            raise ValueError(f"Unknown in-type: {self.in_datatype}. Must be {ALLOWED_DATATYPES}")
        if self.out_datatype not in ALLOWED_DATATYPES:
            raise ValueError(f"Unknown out-type: {self.out_datatype}. Must be {ALLOWED_DATATYPES}")
        if self.rotation_type not in ALLOWED_ROTATIONS:
            raise ValueError(f"Unknown rotation type: {self.rotation_type}. Must be {ALLOWED_ROTATIONS}")

        if isinstance(offset, tuple) or isinstance(offset, list):
            if len(offset) != 3:
                raise ValueError(f"Offset must be a tuple of 3 integers. Got {offset}.")

        transform = transforms.ToTensor()
        self.dataset = datasets.MNIST(root="dataset/MNIST", train=self.MNIST_train,
                                      transform=transform, download=True)
        
        if self.MNIST_train:
            train_size = int(0.8 * len(self.dataset ))
            val_size = len(self.dataset ) - train_size
            train_dataset, val_dataset = random_split(self.dataset , [train_size, val_size],
                                        generator=torch.Generator().manual_seed(self.random_seed))
            if split == "train":
                self.dataset = train_dataset
            elif split == "val":
                self.dataset =  val_dataset
            elif split == "trainval":
                self.dataset = train_dataset + val_dataset
        
        self.rotation_type = rotation_type

        self.caching = caching

        if self.caching:
            CACHE_PATH = f"dataset/MNIST/preloaded_MNIST_{self.split}_{self.in_datatype}_{self.out_datatype}_{self.mode}_{self.rotation_type}_{self.n_alpha}_{self.n_beta}.pkl"
            if os.path.exists(CACHE_PATH):
                with open(CACHE_PATH, "rb") as f:
                    self.preloaded_dataset = pickle.load(f)
                    self.is_preloaded = True
            else:
                # Load all dataset and apply rotation from list if needed
                self.preload()

                # If the rotation is fixe (validation or test), we apply it now
                if rotation_type != "no" and (self.split == "val" or self.split == "test"):
                    self.rotate_from_list(device=compute_device, rotations_list=rotations_list, rotation_type=rotation_type)

                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(self.preloaded_dataset, f)
            
            # We are caching the non-offsetted dataset, so we apply the offset now if needed
            if tuple(self.offset) != (0,0,0):
                    self.apply_offset(offset=self.offset)
            

    def __getitem__(self, index):
        """
        Returns an item from the dataset, possibly preloaded, with the specified projection type.

        Projection types:
        - simple: associate every pixel with an equiangular pixel on the sphere. The image is
                  distorted.
        - equirectangular: project the image onto the sphere using spherical projection.
        """
        if self.is_preloaded:
            return self.preloaded_dataset[index]

        #x_orig is scalar
        #y_orig is label
        x_orig, y_orig = self.dataset[index]

        if self.mode == "simple":
            # Input can be scalar or vector
            if self.in_datatype == "scalar":
                x = stitch_mnist_to_sphere_simple(x_orig, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])
            elif self.in_datatype == "vector":
                u,v = compute_gradients(x_orig)
                x = 1j*stitch_mnist_to_sphere_simple(u, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])\
                    - stitch_mnist_to_sphere_simple(v, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])

            # Output can be label (classification), scalar, or vector
            if self.out_datatype == "label":
                y = y_orig
            elif self.out_datatype == "scalar":
                y = stitch_mnist_to_sphere_simple(x_orig, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])
            elif self.out_datatype == "vector":
                u, v = compute_gradients(x_orig)
                # recall: real vector field to complex scalar -> i(u+iv) = iu - v
                y = 1j*stitch_mnist_to_sphere_simple(u, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])\
                    - stitch_mnist_to_sphere_simple(v, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])

        elif self.mode == "equirectangular":
            # Input can be scalar or vector
            if self.in_datatype == "scalar":
                x = stitch_mnist_to_sphere_proj(x_orig, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])
            elif self.in_datatype == "vector":
                u, v = compute_gradients(x_orig)
                x = 1j* stitch_mnist_to_sphere_proj(u, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])\
                      - stitch_mnist_to_sphere_proj(v, self.n_alpha, self.n_beta, alpha_offset=self.offset[0])

            # TODO: we can save some time if we copy the input to the output
            # Output can be label (classification), scalar, or vector
            if self.out_datatype == "label":
                y = y_orig
            elif self.out_datatype == "scalar":
                y = stitch_mnist_to_sphere_proj(x_orig, self.n_alpha, self.n_beta)
            elif self.out_datatype == "vector":
                u, v = compute_gradients(x_orig)
                # recall: real vector field to complex scalar -> i(u+iv) = iu - v
                y = 1j * stitch_mnist_to_sphere_proj(u, self.n_alpha, self.n_beta)\
                    - stitch_mnist_to_sphere_proj(v, self.n_alpha, self.n_beta)

        elif self.mode == "spherical":
            # Input can be scalar or vector
            if self.in_datatype == "scalar":
                x = stitch_mnist_to_sphere_spherical(x_orig, self.n_alpha, self.n_beta)
            elif self.in_datatype == "vector":
                u, v = compute_gradients(x_orig)
                x = 1j * stitch_mnist_to_sphere_spherical(u, self.n_alpha, self.n_beta) \
                    - stitch_mnist_to_sphere_spherical(v, self.n_alpha, self.n_beta)

            # Output can be label (classification), scalar, or vector
            if self.out_datatype == "label":
                y = y_orig
            elif self.out_datatype == "scalar":
                y = stitch_mnist_to_sphere_spherical(x_orig, self.n_alpha, self.n_beta)
            elif self.out_datatype == "vector":
                u, v = compute_gradients(x_orig)
                y = 1j * stitch_mnist_to_sphere_spherical(u, self.n_alpha, self.n_beta) \
                    - stitch_mnist_to_sphere_spherical(v, self.n_alpha, self.n_beta)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        x = x.to(torch.complex64)
        if self.out_datatype != "label":
            y = y.to(torch.complex64)

        return x, y
    

######################
### Spherical ERA5 ###
######################
    
MAX_WIND = 56
MAX_TEMP = 60
MIN_TEMP = -90
ANGLE_EPS = 0.1

class WeatherDataset(SphericalDataset):
    """
    Custom Wind Dataset where all the data is provided in a 
    Nx2xWxL matrix where N is the number of samples, 2 are the directions (u,v)
    of the wind and W,L are the possible values for longitude and latitude
        
    Masking is executed either at init. or get depending on train/test
    
    mode: geo/spectra
    
    """

    def __init__(self,
                 root,
                 folders=None,
                 train=False,
                 datatype="torch",
                 in_datatype="vector",
                 out_datatype="vector"):
        """
        Args:
        folders: path to folders with numpy datapoints
        train: true if it is a training dataset
        datatype: "torch" or "numpy" to specify the type of data to load. "torch" is preprocessed (legacy).
        in_datatype: "vector" or "scalar" to specify the type of input data
        out_datatype: "vector" to specify the type of output data
        """

        super().__init__(n_alpha=360, n_beta=179,
                            in_datatype=in_datatype, out_datatype=out_datatype)
        
        self.root = root
        self.folders = folders
        self.train = train
        self.datatype = datatype

        ALLOWED_IN_DATATYPES = ["scalar", "vector"]
        ALLOWED_OUT_DATATYPES = ["vector",]
        ALLOWED_DATATYPES = ["torch", "numpy"]

        if self.datatype not in ALLOWED_DATATYPES:
            raise Exception(f"Datatype {self.datatype} not supported. Use 'torch' or 'numpy'.")

        if self.in_datatype not in ALLOWED_IN_DATATYPES or \
                            self.out_datatype not in ALLOWED_OUT_DATATYPES:
            raise Exception(f"Not supported X={self.in_datatype} and Y={self.out_datatype}.")
        
        if folders == None:
            self.idx_to_data = os.listdir(os.path.join(root,"X"))
        else:
            self.idx_to_data = []
            for sf in folders:
                self.idx_to_data.extend([[str(sf),el] for el in os.listdir(os.path.join(root, str(sf), "X"))])
        
        
    def __len__(self):
        return len(self.idx_to_data)

    def __getitem__(self, idx):
        if self.is_preloaded:
            return self.preloaded_dataset[idx]
        else:
            if self.folders == None:
                path_X = os.path.join(self.root,"X",self.idx_to_data[idx])
                path_Y = os.path.join(self.root,"Y",self.idx_to_data[idx])
            else:
                path_X = os.path.join(self.root, self.idx_to_data[idx][0],"X", self.idx_to_data[idx][1])
                path_Y = os.path.join(self.root, self.idx_to_data[idx][0],"Y", self.idx_to_data[idx][1])

            if self.datatype == "torch":
                X = torch.load(path_X)
                Y = torch.load(path_Y)
            elif self.datatype == "numpy":
                X = np.load(path_X)
                Y = np.load(path_Y)

                # Process X with proper cleanup of intermediate tensors
                if self.in_datatype == "vector":
                    X = np.clip(X, -MAX_WIND, MAX_WIND)
                    X = X / MAX_WIND
                    X_tensor = torch.from_numpy(X)
                    del X  # Free numpy array
                    X = torch.view_as_complex(torch.stack([-X_tensor[1], X_tensor[0]], dim=2)).to(torch.complex64)
                    del X_tensor  # Free intermediate tensor
                elif self.in_datatype == "scalar":
                    X = (X - 273.15 - MIN_TEMP)/(MAX_TEMP-MIN_TEMP)
                    X = torch.from_numpy(X)

                # Process Y with proper cleanup of intermediate tensors
                if self.out_datatype == "vector":
                    Y = np.clip(Y, -MAX_WIND, MAX_WIND)
                    Y = Y / MAX_WIND
                    Y_tensor = torch.from_numpy(Y)
                    del Y  # Free numpy array
                    Y = torch.view_as_complex(torch.stack([-Y_tensor[1], Y_tensor[0]], dim=2)).to(torch.complex64)
                    del Y_tensor  # Free intermediate tensor

            return X, Y