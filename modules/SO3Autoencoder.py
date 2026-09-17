import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.nn.functional as F

from tqdm import tqdm
import time

from .SO3fft import S2_to_SO3, SO3_to_S2
from .SO3fft import S2FFT
from .SO3fft import S2iFFT
from .SO3fft import SO3FFT, SO3iFFT

from .SpectralModules import *
from . import precomp
from .SO3lib import normalise_spins
from .architectures import resolve
from .SO3ConvLayer import SO3Conv, SpinSO3Conv

"""
SO3 Autoencoder architectures

Resolutions: order L at a specific layer (same in input/output)
Widths: the number of features at every depth, e.g. [1, 2, 8, 30, 38, 10, 1]
Spins: the spins, e.g. [0, [0,1], [0,1], [0,1], [0,1], [0,1], 0]

"""

class SO3Autoencoder(nn.Module):
    """An equivariant UNet without the skip connections.

    Everything reaching the output passes through the bottleneck, which is
    what makes it an autoencoder rather than a UNet.
    """

    def __init__(self,
                 n_alpha: int,
                 n_beta: int,
                 arch:str=None,
                 layer_style:str=None,
                 fft_method:str="full",
                 resolutions:list=None,
                 widths:list=None,
                 spins:list=None,
                 use_batch_norm:bool=True,
                 use_layer_norm:bool=False,
                 nonlinearity:str=None,
                 wigner_n0:list=None,
                 wigner_n1:list=None,
                 f_wigner:list = None,
                 bdeltas:list=None,
                 bdeltas_prod:list=None,
                 sines:list=None,
                 ):

        """Initialize SO3Autoencoder model.

        Args:
            n_alpha (int): Expected initial/final longitude resolution (number of grid points).
            n_beta (int): Expected initial/final latitude resolution (number of grid points).
            arch (str, optional): "s3cnn" or "swscnn"; see `modules/architectures.py`.
            layer_style (str, optional): The pre-M2 spelling of `arch`. 
                Defaults to "so3".
            fft_method (str, optional): Type of d-matrices to use. Either "full", "delta", 
                or "prod_delta". Defaults to "full".
            resolutions (list, optional): List of resolutions for each layer. The length 
                corresponds to the number of layers. Defaults to None.
            widths (list, optional): List of widths for each layer (number of features). 
                Defaults to None.
            spins (list, optional): List of spins for each layer (0 or 1). Defaults to None.
            use_batch_norm (bool, optional): Whether to use batch normalization. 
                Defaults to True.
            use_layer_norm (bool, optional): Whether to use layer normalization. Defaults to False.
            nonlinearity (str, optional): Type of nonlinearity to use. Defaults to "magnitude".
            wigner_n0 (list, optional): Slice of Wigner d-matrices corresponding to n=0. 
                If None, will be loaded from precomputed files. Defaults to None.
            wigner_n1 (list, optional): Slice of Wigner d-matrices corresponding to n=1. 
                If None, will be loaded from precomputed files. Defaults to None.
            f_wigner (list, optional): Full Wigner d-matrices. If None, will be loaded 
                from precomputed files. Defaults to None.
            bdeltas (list, optional): Black deltas coefficients. Defaults to None.
            bdeltas_prod (list, optional): Black deltas with precomputed products. 
                Defaults to None.
            sines (list, optional): Sampled sines for the delta FFT. Defaults to None.

        Raises:
            ValueError: If last spin is not a single value of 0 or 1.
            ValueError: If widths, spins, and resolutions don't have compatible lengths.
            ValueError: If number of layers is less than 2 or not odd-numbered.
            ValueError: If arch names no known architecture.
            ValueError: If the nonlinearity is one the architecture does not admit.
            ValueError: If fft_method is not valid for the chosen architecture.
        """

        super(SO3Autoencoder, self).__init__()

        ### Load parameters in the class ###

        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.spec = resolve(arch, layer_style)
        self.arch = self.spec.name
        self.fft_method = fft_method
        self.resolutions = resolutions
        self.widths = widths
        self.spins = spins
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        # Which nonlinearities are admissible is a property of the architecture,
        # not a free choice: see `modules/architectures.py`.
        self.nonlinearity_type = self.spec.check_sigma(
            self.spec.sigma if nonlinearity is None else nonlinearity)

        # `spins` accepts 0, [0] or [0,1] per layer; normalise so the rest of the
        # constructor can assume lists.
        self.spins = normalise_spins(self.spins)

        if len(self.spins[-1]) != 1:
            raise ValueError("Last spin must be a single value, got {}".format(self.spins[-1]))
        self.output = "scalar" if self.spins[-1][0] == 0 else "vector"
        
        ### Sanity check ###
        # Example of resolutions: [179, 100, 50, 100, 179]
        # Example of widths: [1, 2, 8, 30, 10, 1]
        # Example of spins: [0, [0,1], [0,1], [0,1], [0,1], [0,1], 0]

        # In this case the layers are computed as:
        # Layer 1: 1 (S=0, L=179) -> 2 (S=[0,1], L=179)
        # Layer 2: 2 (S=[0,1], L=100) -> 8 (S=[0,1], L=100)
        # Layer 3: 8 (S=[0,1], L=50) -> 30 (S=[0,1], L=50)
        # Layer 4: 30 (S=[0,1], L=38) -> 38 (S=[0,1], L=38)
        # Layer 5: 38 (S=[0,1], L=10) -> 10 (S=[0,1], L=10)
        # Layer 6: 10 (S=[0,1], L=1) -> 1 (S=0, L=1)

        if len(self.resolutions) != len(self.widths)-1 or len(self.widths) != len(self.spins):
            raise ValueError("Widths and spins must have same length, and resolution must be one fewer element")

        if len(self.resolutions) < 2 or len(self.resolutions)% 2 == 0:
            raise ValueError("Layers must be at least two elements and be odd-numbered")
        

        # The group-filter path (scnn) needs SO(3) pooling, which only the flat
        # SO3CNN has so far; see `modules/architectures.py`.
        if self.spec.filters == "so3":
            raise NotImplementedError(
                f"{type(self).__name__} has no scnn path yet: pooling on the "
                f"group is not implemented here. Use SO3CNN for scnn.")
        
        if self.spec.lifts:
            if self.fft_method not in ["full", "delta"]:
                raise ValueError("Method must be 'full' or 'delta' for SO3 layers, not {}".format(self.fft_method))

        ### Load the precomputed coefficients, if these have not been passed as an argument ###

        self.needs_0_spin = any( 0 in sublist for sublist in self.spins)
        self.needs_1_spin = any(1 in sublist for sublist in self.spins)

        # Initialization of the Wigner d-matrices for n=0
        if self.needs_0_spin:
            if wigner_n0 == None:
                # If no Wigner d-matrices are provided, load them from the precomputed file to the DEVICE
                self.wigner_n0 = \
                    precomp.load("SLICE_WIGNER_N0", n_beta, max(self.resolutions))
            else:
                # If Wigner d-matrices are provided, make sure they are on the DEVICE
                self.wigner_n0 = [m.requires_grad_(False) for m in wigner_n0]

            # Register buffer for coefficients but exclude them from state_dict
            for i,m in enumerate(self.wigner_n0): 
                self.register_buffer("wigner_n0_"+str(i), m, persistent=False)
        else:
            # If no Wigner d-matrices are needed, set the variable to None
            self.wigner_n0 = None

        # Initialization of the Wigner d-matrices for n=1
        if self.needs_1_spin: 
            if wigner_n1 == None:
                # If no Wigner d-matrices are provided, load them from the precomputed file to the DEVICE
                self.wigner_n1 =\
                    precomp.load("SLICE_WIGNER_N1", n_beta, max(self.resolutions))
            else:
                # If Wigner d-matrices are provided, make sure they are on the DEVICE
                self.wigner_n1 = [m.requires_grad_(False) for m in wigner_n1]
            
            # Register buffer for coefficients but exclude them from state_dict
            for i,m in enumerate(self.wigner_n1):
                self.register_buffer("wigner_n1_"+str(i), m, persistent=False)
        else:
            self.wigner_n1 = None
        
        # Initialization of the full Wigner d-matrices or bdelta coefficients, depending on the method
        # Either one of those are required for the full SO(3) convolutions. Not needed for the spins method
        if self.spec.lifts:
            if self.fft_method == "delta":
                #TODO Re-implement this method
                # Coefficient sharing with rotator not possible
                pass 
            elif self.fft_method == "full":
                if f_wigner == None:
                    # If no Wigner d-matrices are provided, load them from the precomputed file to the DEVICE
                    self.f_wigner = precomp.load("FULL_WIGNER", n_beta, max(self.resolutions))
                else:
                    # If Wigner d-matrices are provided, make sure they are on the DEVICE
                    self.f_wigner = [m.requires_grad_(False) for m in f_wigner]
                
                # Register buffer for coefficients but exclude them from state_dict
                self.f_wigner = torch.nested.nested_tensor(self.f_wigner)
                self.register_buffer("wigner_full",self.f_wigner, persistent=False)

        ### Initialize the module lists for the layers ###
        
        self.convolutions_down = nn.ModuleList()
        self.FFTs_down = nn.ModuleList()
        self.iFFTs_down = nn.ModuleList()
        self.nonlinearities_down = nn.ModuleList()
        self.dropouts_down = nn.ModuleList()
        self.spatial_smoothing_down = nn.ModuleList()
        if self.use_batch_norm:
            self.batchnorm_down = nn.ModuleList()
        if self.use_layer_norm:
            self.layernorm_down = nn.ModuleList()
        
        self.convolutions_up = nn.ModuleList()
        self.FFTs_up = nn.ModuleList()
        self.iFFTs_up = nn.ModuleList()
        self.nonlinearities_up = nn.ModuleList()
        self.dropouts_up = nn.ModuleList()
        self.spatial_smoothing_up = nn.ModuleList()
        if self.use_batch_norm:
            self.batchnorm_up = nn.ModuleList()
        if self.use_layer_norm:
            self.layernorm_up = nn.ModuleList()

        ### Configure the layers ###

        #Layers are made of
        # - SO(3) convolution S2 -> SO(3)
        # - Inverse SO(3) FFT
        # - Nonlinearity
        # - Dropout
        # - Spatial Smoothing
        # - S2 FFT (since we have the smoothing in the spatial domain)
        # - No skip connections in simple encoder-decoder autoencoder

        self.fs_down = []
        self.fs_up = []
        for i in range(len(self.resolutions)//2):
            self.fs_down.append({'f_in':self.widths[i],
                                 'f_out':self.widths[i+1],
                                 'in_spins': self.spins[i],
                                 'out_spins': self.spins[i+1],
                                 'L':self.resolutions[i],
                                 'a':(self.resolutions[i]+1)*2})
        
        for i in range(len(self.resolutions)//2, len(self.resolutions)):
            # Symmetric encoder-decoder without skip connections
            self.fs_up.append({'f_in':self.widths[i],
                               'f_out':self.widths[i+1],
                               'in_spins': self.spins[i],
                               'out_spins': self.spins[i+1],
                               'L':self.resolutions[i],
                               'a':(self.resolutions[i]+1)*2})
        
        ### Initialize the layers ###

        # DOWN layers
        for i in range(len(self.fs_down)):

            #SO(3) convolution is n-agnostic
            if self.spec.lifts:
                self.convolutions_down.append(SO3Conv(
                    nfeature_in=self.fs_down[i]['f_in'],
                    nfeature_out=self.fs_down[i]['f_out'],
                    L=self.fs_down[i]['L'],
                ))
            else:
                self.convolutions_down.append(SpinSO3Conv(
                    nfeature_in=self.fs_down[i]['f_in'],
                    nfeature_out=self.fs_down[i]['f_out'],
                    L=self.fs_down[i]['L'],
                ))

            if self.spec.lifts:
                self.iFFTs_down.append(SO3iFFT(
                    L= self.fs_down[i]['L'],
                    n_alpha = self.fs_down[i]['a'],
                    n_beta = self.n_beta,
                    n_gamma = self.fs_down[i]['a'],
                ))
            else:
                self.iFFTs_down.append(S2iFFT(
                    L= self.fs_down[i]['L'],
                    n_alpha = self.fs_down[i]['a'],
                    n_beta = self.n_beta,
                    n = self.fs_down[i]['out_spins'],
                ))
            
            if self.spec.lifts:
                if self.use_batch_norm:
                    self.batchnorm_down.append(
                        FullSpectralBatchNormalization(spins = self.fs_down[i]['out_spins'],
                                                    nfeatures = self.fs_down[i]['f_out']))
                if self.use_layer_norm:
                    self.layernorm_down.append(
                        FullSpectralLayerNormalization(n = self.fs_down[i]['out_spins'], nfeatures = self.fs_down[i]['f_out']))
                if self.nonlinearity_type == "relu":
                    self.nonlinearities_down.append(WeightedReLU())
                elif self.nonlinearity_type == "leaky_relu":
                    self.nonlinearities_down.append(WeightedLeakyReLU())
                self.spatial_smoothing_down.append(SpatialSmoothing(n=self.fs_down[i]['out_spins']))
            else:
                if self.use_batch_norm:
                    self.batchnorm_down.append(
                        SpinSpectralBatchNormalization(nfeatures = self.fs_down[i]['f_out'],
                                                    n = self.fs_down[i]['out_spins']))
                if self.use_layer_norm:
                    self.layernorm_down.append(
                        SpinSpectralLayerNormalization(nfeatures = self.fs_down[i]['f_out'],
                                                    n = self.fs_down[i]['out_spins']))
                if self.nonlinearity_type == "phase_collapse":
                    self.nonlinearities_down.append(PhaseCollapseNonlinearity(
                                                    n=self.fs_down[i]['out_spins'],
                                                    nfeatures=self.fs_down[i]['f_out']))
                elif self.nonlinearity_type == "magnitude":
                    self.nonlinearities_down.append(
                        MagnitudeNonlinearity(n=self.fs_down[i]['out_spins']))

            self.dropouts_down.append(ComplexDropout(p=0.2))

            self.FFTs_down.append(S2FFT(
                L=self.fs_down[i]['L'],
                n=self.fs_down[i]['out_spins'],
            ))

        # UP layers
        for i in range(len(self.fs_up)):
            if self.spec.lifts:
                self.convolutions_up.append(SO3Conv(
                    nfeature_in=self.fs_up[i]['f_in'],
                    nfeature_out=self.fs_up[i]['f_out'],
                    L= self.fs_up[i]['L'],
                ))
            else:
                self.convolutions_up.append(SpinSO3Conv(
                    nfeature_in=self.fs_up[i]['f_in'],
                    nfeature_out=self.fs_up[i]['f_out'],
                    L= self.fs_up[i]['L'],
                ))

            # `forward` skips everything after the convolution on the last up
            # layer, so building it there would only create dead parameters.
            if i == len(self.fs_up) - 1:
                continue

            if self.spec.lifts:
                self.iFFTs_up.append(SO3iFFT(
                    L= self.fs_up[i]['L'],
                    n_alpha = self.fs_up[i]['a'],
                    n_beta = self.n_beta,
                    n_gamma = self.fs_up[i]['a'],
                ))
            else:
                self.iFFTs_up.append(S2iFFT(
                    L= self.fs_up[i]['L'],
                    n_alpha = self.fs_up[i]['a'],
                    n_beta = self.n_beta,
                    n = self.fs_up[i]['out_spins'],
                ))
            
            if self.spec.lifts:
                if self.use_batch_norm:
                    self.batchnorm_up.append(
                        FullSpectralBatchNormalization(spins = self.fs_up[i]['out_spins'],
                                                    nfeatures = self.fs_up[i]['f_out']))
                if self.use_layer_norm:
                    self.layernorm_up.append(
                        FullSpectralLayerNormalization(n = self.fs_up[i]['out_spins'], nfeatures = self.fs_up[i]['f_out']))
                if self.nonlinearity_type == "relu":
                    self.nonlinearities_up.append(WeightedReLU())
                elif self.nonlinearity_type == "leaky_relu":
                    self.nonlinearities_up.append(WeightedLeakyReLU())
                self.spatial_smoothing_up.append(SpatialSmoothing(n=self.fs_up[i]['out_spins']))
            else:
                if self.use_batch_norm:
                    self.batchnorm_up.append(
                        SpinSpectralBatchNormalization(nfeatures = self.fs_up[i]['f_out'],
                                                    n = self.fs_up[i]['out_spins']))
                if self.use_layer_norm:
                    self.layernorm_up.append(
                        SpinSpectralLayerNormalization(nfeatures = self.fs_up[i]['f_out'],
                                                    n = self.fs_up[i]['out_spins']))
                if self.nonlinearity_type == "phase_collapse":
                    self.nonlinearities_up.append(PhaseCollapseNonlinearity(
                                                    n=self.fs_up[i]['out_spins'],
                                                    nfeatures=self.fs_up[i]['f_out']))
                elif self.nonlinearity_type == "magnitude":
                    self.nonlinearities_up.append(
                        MagnitudeNonlinearity(n=self.fs_up[i]['out_spins']))

            self.dropouts_up.append(ComplexDropout(p=0.2))

            self.FFTs_up.append(S2FFT(
                L=self.fs_up[i]['L'],
                n=self.fs_up[i]['out_spins'],
            ))

        # Input and output transforms
        self.s2fft = S2FFT(L=self.fs_down[0]['L'], n=self.spins[0])
        self.s2ifft = S2iFFT(L=self.fs_up[-1]['L'], n_alpha=self.n_alpha, n_beta=self.n_beta, n=self.spins[-1])
        
        # Only the last layer needs to convert from SO(3) to S2, which is equivalent to a spectral smoothing
        if self.spec.lifts:
            self.so3_to_s2 = SO3_to_S2(L=self.resolutions[-1], n=self.spins[-1])

    def get_buffered_slice_wigner(self, n=1):
        """
        Return the sliced Wigner D matrix list on the current device.
        This is used for the S2FFT and S2iFFT layers, and is passed to each call independently if the spins are needed.
        Thus it returns None when the spins are not needed, or a list of buffers when they are needed.
        """
        if n==1:
            if self.needs_1_spin:
                return [self.get_buffer("wigner_n1_"+str(i)) for i in range(len(self.wigner_n1))]
            else:
                return None
        elif n==0:
            if self.needs_0_spin:
                return [self.get_buffer("wigner_n0_"+str(i)) for i in range(len(self.wigner_n0))]
            else:
                return None
    
    def get_buffered_full_wigner(self):
        """
        Return the Full Wigner D matrix list on the current device.
        This is used for the SO3iFFT layers, and is passed to each call.
        """
        return self.get_buffer("wigner_full")
    
    def unpool(self, x, target_L):
        if self.spec.lifts:
            current_L = len(x)
            for i in range(current_L, target_L+1):
                x.append(torch.zeros((x[0].shape[0], x[0].shape[1], i*2+1, i*2+1), device=x[0].device))
        elif not self.spec.lifts:
            x = F.pad(x, (0, target_L*(target_L+2)+1 - x.shape[-1]))
        return x

    def forward(self, x):
        batch_size = x.size(0)

        #If the input is a scalar, it might be encoded as a float32 tensor.
        #We convert it to a complex tensor to be able to use the SO3 convolutions.
        if x.dtype == torch.float32:
            x = torch.view_as_complex(x)
        
        #If the input has only one feature, the feature dimension might be missing.
        if len(x.shape) < 4:
            x = x.reshape(batch_size, 1, self.n_alpha, self.n_beta)

        x = self.s2fft(x,
                       d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                       d_matrices_n1=self.get_buffered_slice_wigner(n=1))

        # DOWN layers (encoder)
        for i in range(len(self.convolutions_down)):
            #print(f"Down layer {i}")
            #print(self.fs_down[i])
            x = self.convolutions_down[i](x) #S2 -> S2/SO(3)

            if self.use_batch_norm:
                x = self.batchnorm_down[i](x) # Batch normalization
            if self.use_layer_norm:
                x = self.layernorm_down[i](x) # Layer normalization

            if self.spec.lifts:
                x = self.iFFTs_down[i](x, d_matrices=self.get_buffered_full_wigner())
            else:
                x = self.iFFTs_down[i](x,
                                   d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                                   d_matrices_n1=self.get_buffered_slice_wigner(n=1))

            x = self.nonlinearities_down[i](x) # Nonlinearity/activation function in the spatial domain
            x = self.dropouts_down[i](x) # Space dropout

            if self.spec.lifts:
                x = self.spatial_smoothing_down[i](x) # [alpha, beta, gamma] -> [alpha, beta]

            # [alpha, beta] -> [l,m]
            x = self.FFTs_down[i](x,
                                  d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                                  d_matrices_n1=self.get_buffered_slice_wigner(n=1))
            
            # Spectral pooling: truncate high-frequency coefficients for downsampling
            if i < len(self.convolutions_down)-1:
                x = x[...,:self.fs_down[i+1]["L"]*(self.fs_down[i+1]["L"]+2)+1]

        # At bottleneck, convert from SO(3) to S2 if needed
        # NOTE: This conversion happens AFTER all UP layers complete, not before
        # (see below after UP layers)
        
        # UP layers (decoder) - no skip connections
        for i in range(len(self.convolutions_up)):
            #print(f"Up layer {i}")
            #print(self.fs_up[i])
            x = F.pad(x, (0, self.fs_up[i]['L']*(self.fs_up[i]['L']+2)+1 - x.shape[-1]))
            # Convolution does not mix frequencies
            x = self.convolutions_up[i](x) #S2 -> S2/SO(3)

            # Unpooling to increase resolution by padding with zeros in the spectral domain
            #x = self.unpool(x, target_L=self.fs_up[i]['L']) 

            
            # Not on the last layer
            if i < len(self.convolutions_up)-1:
                if self.use_batch_norm:
                    x = self.batchnorm_up[i](x) # Batch normalization
                if self.use_layer_norm:
                    x = self.layernorm_up[i](x) # Layer normalization

                if self.spec.lifts:
                    x = self.iFFTs_up[i](x, d_matrices=self.get_buffered_full_wigner())
                else:
                    x = self.iFFTs_up[i](x,
                                        d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                                        d_matrices_n1=self.get_buffered_slice_wigner(n=1))

                x = self.nonlinearities_up[i](x) # Nonlinearity/activation function in the spatial domain
                x = self.dropouts_up[i](x) # space dropout
                
                if self.spec.lifts:
                    x = self.spatial_smoothing_up[i](x) # [alpha, beta, gamma] -> [alpha, beta]
                x = self.FFTs_up[i](x,
                                    d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                                    d_matrices_n1=self.get_buffered_slice_wigner(n=1))

        # At this point, if x is in the SO(3) domain, we slice to S2
        # This is done AFTER all UP layers, consistent with SO3UNet
        if self.spec.lifts:
            x = self.so3_to_s2(x)

        # Final inverse FFT to spatial domain
        x = self.s2ifft(x,
                        d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                        d_matrices_n1=self.get_buffered_slice_wigner(n=1))

        x = x.reshape(batch_size, self.n_alpha, self.n_beta)
        
        return x