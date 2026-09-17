import sys
sys.path.append("../")

import torch
import torch.nn as nn

from .SO3fft import S2_to_SO3, SO3_to_S2
from .SO3fft import S2FFT
from .SO3fft import S2iFFT
from .SO3fft import SO3FFT, SO3iFFT

from . import precomp
from .SO3lib import normalise_spins
from .SO3ConvLayer import SO3Conv, SpinSO3Conv

from .SpectralModules import ComplexDropout, MagnitudeNonlinearity, PhaseCollapseNonlinearity, WeightedLeakyReLU, \
    SpatialSmoothing, SphericalBatchNormalization, SpinSpectralBatchNormalization, SpinSpectralLayerNormalization, \
    FullSpectralLayerNormalization

class SO3MNISTClassifier(nn.Module):

    def __init__(self,
                 n_alpha: int,
                 n_beta: int,
                 layer_style:str="so3",
                 fft_method:str="full",
                 resolutions:list=None,
                 widths:list=None,
                 spins:list=None,
                 head:bool=True,
                 nonlinearity:str="magnitude",
                 use_batch_norm:bool=True,
                 use_layer_norm:bool=False,
                 wigner_n0:list=None,
                 wigner_n1:list=None,
                 f_wigner:list = None,
                 bdeltas:list=None,
                 bdeltas_prod:list=None,
                 sines:list=None):
        
        """
        SO3 MNIST Classifier initialization.
        
        Args:
            n_alpha (int): Expected initial/final longitude resolution (number of grid points).
            n_beta (int): Expected initial/final latitude resolution (number of grid points).
            layer_style (str, optional): Type of layers to use, either "so3" or "spins". 
                Defaults to "so3".
            fft_method (str, optional): FFT implementation to use. Defaults to "full".
            resolutions (list, optional): List of resolutions for each layer. Length is the 
                number of layers. Defaults to None.
            widths (list, optional): List of widths for each layer (number of features). 
                Defaults to None.
            spins (list, optional): List of spins for each layer (0 or 1). Defaults to None.
            head (bool, optional): If True, adds a final linear layer to the model. 
                Defaults to True.
            use_batch_norm (bool, optional): Whether to use batch normalization. 
            use_layer_norm (bool, optional): Whether to use layer normalization.
            wigner_n0 (list, optional): Slice of Wigner d-matrices corresponding to n=0. 
                Defaults to None.
            wigner_n1 (list, optional): Slice of Wigner d-matrices corresponding to n=1. 
                Defaults to None.
            f_wigner (list, optional): Full Wigner d-matrices. Defaults to None.
            bdeltas (list, optional): Black deltas coefficients. Defaults to None.
            bdeltas_prod (list, optional): Black deltas with precomputed products. 
                Defaults to None.
            sines (list, optional): Sampled sines for the delta FFT. Defaults to None.
        """

        super(SO3MNISTClassifier, self).__init__()
        
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.layer_style = layer_style
        self.fft_method = fft_method
        self.resolutions = resolutions
        self.widths = widths
        self.spins = spins
        self.head = head
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.nonlinearity_type = nonlinearity

        # `spins` accepts 0, [0] or [0,1] per layer; normalise so the rest of the
        # constructor can assume lists.
        self.spins = normalise_spins(self.spins)

        if len(self.spins[-1]) != 1:
            raise ValueError("Last spin must be a single value, got {}".format(self.spins[-1]))
        self.output = "scalar" if self.spins[-1][0] == 0 else "vector"
        
        ### Load the precomputed coefficients, if these have not been passed as an argument ###

        self.needs_0_spin = any( 0 in sublist for sublist in self.spins)
        self.needs_1_spin = any(1 in sublist for sublist in self.spins)

        # Initialization of the Wigner d-matrices for n=0
        if self.needs_0_spin:
            if wigner_n0 == None:
                # If no Wigner d-matrices are provided, load them from the precomputed file
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
                # If no Wigner d-matrices are provided, load them from the precomputed file
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
        
        if self.layer_style == "so3":
            if self.fft_method == "delta":
                #TODO Re-implement this method
                # Coefficient sharing with rotator not possible
                pass 
            elif self.fft_method == "full":
                if f_wigner == None:
                    # If no Wigner d-matrices are provided, load them from the precomputed file
                    self.f_wigner = precomp.load("FULL_WIGNER", n_beta, max(self.resolutions))
                else:
                    # If Wigner d-matrices are provided, make sure they are on the DEVICE
                    self.f_wigner = [m.requires_grad_(False) for m in f_wigner]
                
                # Register buffer for coefficients but exclude them from state_dict
                self.f_wigner = torch.nested.nested_tensor(self.f_wigner)
                self.register_buffer("wigner_full",self.f_wigner, persistent=False)

        self.layers = []
        for i in range(len(self.widths)-1):
            self.layers.append({
                'f_in': self.widths[i],
                'f_out': self.widths[i+1],
                'in_spins': self.spins[i],
                'out_spins': self.spins[i+1],
                'L_in': self.resolutions[i],
                'L_out': self.resolutions[i+1],
                'a': (self.resolutions[i]+1)*2})
        
        self.convolutions = nn.ModuleList()
        self.FFTs = nn.ModuleList()
        self.iFFTs = nn.ModuleList()
        self.nonlinearities = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.spatial_smoothing = nn.ModuleList()
        if self.use_batch_norm:
            self.batchnorm = nn.ModuleList()
        if self.use_layer_norm:
            self.layernorm = nn.ModuleList()

        # LAYERS
        for i in range(len(self.layers)):
            if self.layer_style == "so3":
                self.convolutions.append(SO3Conv(
                    nfeature_in=self.layers[i]['f_in'],
                    nfeature_out=self.layers[i]['f_out'],
                    L=self.layers[i]['L_in'],
                ))
            elif self.layer_style == "spins":
                self.convolutions.append(SpinSO3Conv(
                    nfeature_in=self.layers[i]['f_in'],
                    nfeature_out=self.layers[i]['f_out'],
                    L=self.layers[i]['L_in'],
                ))

            if self.layer_style == "so3":
                self.iFFTs.append(SO3iFFT(
                    L=self.layers[i]['L_in'],
                    n_alpha = self.layers[i]['a'],
                    n_beta = self.n_beta,
                    n_gamma = self.layers[i]['a'],
                ))
            elif self.layer_style == "spins":
                self.iFFTs.append(S2iFFT(
                    L=self.layers[i]['L_in'],
                    n_alpha = self.layers[i]['a'],
                    n_beta = self.n_beta,
                    n = self.layers[i]['out_spins'],
                ))
            
            self.dropouts.append(ComplexDropout(p=0.2))

            if self.layer_style == "so3":
                self.nonlinearities.append(WeightedLeakyReLU())
                self.spatial_smoothing.append(SpatialSmoothing(n=self.layers[i]['out_spins']))
                if self.use_batch_norm:
                    self.batchnorm.append(
                    SphericalBatchNormalization(nfeatures = self.layers[i]['f_out'], n_beta=self.n_beta,
                                                n = self.layers[i]['out_spins'], dimensions=3))
                if self.use_layer_norm:
                    self.layernorm.append(
                    FullSpectralLayerNormalization(n = self.layers[i]['out_spins'], nfeatures = self.layers[i]['f_out']))
            elif self.layer_style == "spins":
                if self.nonlinearity_type == "magnitude":
                    self.nonlinearities.append(MagnitudeNonlinearity(n=self.layers[i]['out_spins']))
                elif self.nonlinearity_type == "phase_collapse":
                    self.nonlinearities.append(
                        PhaseCollapseNonlinearity(n=self.layers[i]['out_spins'],
                                                nfeatures=self.layers[i]['f_out']))
                if self.use_batch_norm:
                    self.batchnorm.append(
                    SpinSpectralBatchNormalization(nfeatures = self.layers[i]['f_out'],
                                                    n = self.layers[i]['out_spins']))
                if self.use_layer_norm:
                    self.layernorm.append(
                    SpinSpectralLayerNormalization(nfeatures = self.layers[i]['f_out'],
                                                    n = self.layers[i]['out_spins']))

            self.FFTs.append(S2FFT(
                L=self.layers[i]['L_out'],
                n=self.layers[i]['out_spins'],
            ))

        # Input transform
        self.s2fft = S2FFT(L=self.layers[0]['L_in'], n=self.layers[0]['in_spins'])

        if self.head:
            self.final_layer = nn.Linear(self.widths[-1], 10)

    def get_buffered_slice_wigner(self, n=1):
        """
        Returns the wigner d-matrices stored in the model.
        It is used for the FFT and iFFT layers (S2) in each forward pass.
        None if the spins are not part of the model.
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
    
    def randomize_all_weights(self):
        """
        Randomizes all weights in the model. This is used for testing purposes, for example when
        testing the model's equivariance.
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)

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

        # layers
        for i in range(len(self.layers)):
            x = self.convolutions[i](x) #S2 -> SO(3)

            if self.use_batch_norm:
                x = self.batchnorm[i](x) # Batch normalization
            if self.use_layer_norm:
                x = self.layernorm[i](x) # Layer normalization

            if self.layer_style == "so3":
                x = self.iFFTs[i](x, d_matrices=self.get_buffered_full_wigner())
            else:
                x = self.iFFTs[i](x,
                                   d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                                   d_matrices_n1=self.get_buffered_slice_wigner(n=1))
            
            x = self.nonlinearities[i](x) # Nonlinearity/activation function in the spatial domain
            x = self.dropouts[i](x) # Space dropout

            if self.layer_style == "so3":
                x = self.spatial_smoothing[i](x) # [alpha, beta, gamma] -> [alpha, beta]

            # [alpha, beta] -> [l,m]
            x = self.FFTs[i](x,
                            d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                            d_matrices_n1=self.get_buffered_slice_wigner(n=1))
            
        #Final layer should only have one order, which makes the signal invariant
        if x.shape[-1] != 1:
            raise ValueError("Final output should have only one order (L=0), got shape {}".format(x.shape))
            
        x = torch.abs(x)
        x = x.reshape(batch_size, -1)
        
        # If self.head is True, we add a final linear layer
        if self.head:
            x = self.final_layer(x)

        return x