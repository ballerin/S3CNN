import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.nn.functional as F

from tqdm import tqdm

from .SO3fft import S2_to_SO3, SO3_to_S2
from .SO3fft import S2FFT
from .SO3fft import S2iFFT
from .SO3fft import SO3FFT, SO3iFFT

from . import precomp
from .SO3lib import normalise_spins
from .SO3ConvLayer import SO3Conv, SpinSO3Conv, SO3toSO3Conv

from .SpectralModules import ComplexDropout, MagnitudeNonlinearity, WeightedLeakyReLU, \
    SpatialSmoothing, SpinSpectralBatchNormalization, FullSpectralBatchNormalization, \
    PhaseCollapseNonlinearity, WeightedReLU, \
    SpinSpectralLayerNormalization, FullSpectralLayerNormalization

"""
SO3CNN: A flat sequential SO(3) convolutional network for ablation studies.

This architecture is like the UNet but without deeper layers (no pooling/unpooling
and no skip connections). All convolution layers operate at the same resolution,
making it a simple sequential stack of SO(3) equivariant convolutional layers.

INPUT
  | (layer 1)
  v
DATA (L)
  | (layer 2)
  v
DATA (L)
  | ...
  v
DATA (L)
  | (layer N)
  v
OUTPUT

Resolutions: a single resolution L used throughout
Widths: the number of features at every layer, e.g. [1, 2, 8, 30, 8, 1]
Spins: the spins at every layer, e.g. [0, [0,1], [0,1], [0,1], [0,1], 0]

"""

class SO3CNN(nn.Module):

    def __init__(self,
                 n_alpha: int,
                 n_beta: int,
                 layer_style:str="so3",
                 fft_method:str="full",
                 resolution:int=None,
                 widths:list=None,
                 spins:list=None,
                 use_batch_norm:bool=True,
                 use_layer_norm:bool=False,
                 nonlinearity:str="magnitude",
                 wigner_n0:list=None,
                 wigner_n1:list=None,
                 f_wigner:list = None,
                 bdeltas:list=None,
                 bdeltas_prod:list=None,
                 sines:list=None,
                 ):

        """Initialize SO3CNN model.

        This is a flat sequential model without pooling/unpooling or skip connections.
        All layers operate at the same resolution, serving as an ablation baseline
        against the UNet architecture.

        Args:
            n_alpha (int): Expected initial/final longitude resolution (number of grid points).
            n_beta (int): Expected initial/final latitude resolution (number of grid points).
            layer_style (str, optional): Type of layers to use. One of "so3", "spins", or
                "full_so3". "full_so3" keeps the signal on SO(3) (no SpatialSmoothing) across
                every intermediate layer, smoothing back down to S2 only after the last layer.
                Defaults to "so3".
            fft_method (str, optional): Type of d-matrices to use. Either "full", "delta", 
                or "prod_delta". Defaults to "full".
            resolution (int, optional): Single resolution L used throughout all layers. 
                Defaults to None.
            widths (list, optional): List of widths for each layer (number of features). 
                Length is N+1 where N is the number of convolution layers.
                Defaults to None.
            spins (list, optional): List of spins for each layer (0 or 1). Same length as widths.
                Defaults to None.
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
            ValueError: If widths and spins don't have the same length.
            ValueError: If number of layers is less than 2.
            ValueError: If layer_style is not "so3" or "spins".
            ValueError: If fft_method is not valid for the chosen layer_style.
        """

        super(SO3CNN, self).__init__()

        ALLOWED_NONLINEARITIES = ["phase_collapse", "magnitude", "relu", "leaky_relu"]

        ### Load paramenters in the class ###

        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.layer_style = layer_style
        self.fft_method = fft_method
        self.resolutions = resolution
        self.widths = widths
        self.spins = spins
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.nonlinearity_type = nonlinearity

        if self.nonlinearity_type not in ALLOWED_NONLINEARITIES:
            raise ValueError(f"Nonlinearity must be one of {ALLOWED_NONLINEARITIES}, got {self.nonlinearity_type}")

        # `spins` accepts 0, [0] or [0,1] per layer; normalise so the rest of the
        # constructor can assume lists.
        self.spins = normalise_spins(self.spins)

        if len(self.spins[-1]) != 1:
            raise ValueError("Last spin must be a single value, got {}".format(self.spins[-1]))
        self.output = "scalar" if self.spins[-1][0] == 0 else "vector"
        
        ### Sanity check ###
        # Example of resolution: 179
        # Example of widths: [1, 2, 8, 30, 8, 1]
        # Example of spins: [0, [0,1], [0,1], [0,1], [0,1], 0]

        # In this case the layers are computed as:
        # Layer 1: 1 (S=0, L=179) -> 2 (S=[0,1], L=179)
        # Layer 2: 2 (S=[0,1], L=179) -> 8 (S=[0,1], L=179)
        # Layer 3: 8 (S=[0,1], L=179) -> 30 (S=[0,1], L=179)
        # Layer 4: 30 (S=[0,1], L=179) -> 8 (S=[0,1], L=179)
        # Layer 5: 8 (S=[0,1], L=179) -> 1 (S=0, L=179)

        if len(self.widths) != len(self.spins):
            raise ValueError("Widths and spins must have same length")

        if len(self.widths) < 2:
            raise ValueError("Must have at least 2 elements in widths (input and output)")

        if self.resolutions is None:
            raise ValueError("Resolution must be specified")
        
        if self.layer_style not in ["so3", "spins", "full_so3"]:
            raise ValueError("Layer style must be 'so3', 'spins' or 'full_so3', not {}".format(self.layer_style))

        if self.layer_style in ["so3", "full_so3"]:
            if self.fft_method not in ["full", "delta"]:
                raise ValueError("Method must be 'full' or 'delta' for SO3 layers, not {}".format(self.fft_method))

        ### Load the precomputed coefficients, if these have not been passed as an argument ###

        self.needs_0_spin = any( 0 in sublist for sublist in self.spins)
        self.needs_1_spin = any(1 in sublist for sublist in self.spins)

        # Initialization of the Wigner d-matrices for n=0
        if self.needs_0_spin:
            if wigner_n0 == None:
                self.wigner_n0 = \
                    precomp.load("SLICE_WIGNER_N0", n_beta, self.resolutions)
            else:
                self.wigner_n0 = [m.requires_grad_(False) for m in wigner_n0]

            for i,m in enumerate(self.wigner_n0): 
                self.register_buffer("wigner_n0_"+str(i), m, persistent=False)
        else:
            self.wigner_n0 = None

        # Initialization of the Wigner d-matrices for n=1
        if self.needs_1_spin: 
            if wigner_n1 == None:
                self.wigner_n1 =\
                    precomp.load("SLICE_WIGNER_N1", n_beta, self.resolutions)
            else:
                self.wigner_n1 = [m.requires_grad_(False) for m in wigner_n1]
            
            for i,m in enumerate(self.wigner_n1):
                self.register_buffer("wigner_n1_"+str(i), m, persistent=False)
        else:
            self.wigner_n1 = None
        
        # Initialization of the full Wigner d-matrices or bdelta coefficients
        if self.layer_style in ["so3", "full_so3"]:
            if self.fft_method == "delta":
                #TODO Re-implement this method
                pass 
            elif self.fft_method == "full":
                if f_wigner == None:
                    self.f_wigner = precomp.load("FULL_WIGNER", n_beta, self.resolutions)
                else:
                    self.f_wigner = [m.requires_grad_(False) for m in f_wigner]
                
                self.f_wigner = torch.nested.nested_tensor(self.f_wigner)
                self.register_buffer("wigner_full",self.f_wigner, persistent=False)

        ### Initialize the module lists for the layers ###
        
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

        ### Configure the layers ###

        # All layers operate at the same resolution
        L = self.resolutions
        a = (L + 1) * 2

        self.n_layers = len(self.widths) - 1
        self.fs = []
        for i in range(self.n_layers):
            self.fs.append({'f_in':self.widths[i],
                            'f_out':self.widths[i+1],
                            'in_spins': self.spins[i],
                            'out_spins': self.spins[i+1],
                            'L':L,
                            'a':a})
        
        ### Initialize the layers ###

        for i in range(self.n_layers):

            if self.layer_style == "so3":
                self.convolutions.append(SO3Conv(
                    nfeature_in=self.fs[i]['f_in'],
                    nfeature_out=self.fs[i]['f_out'],
                    L=self.fs[i]['L'],
                ))
            elif self.layer_style == "full_so3":
                # Only the first layer lifts from S2 (vector) to SO(3) (matrix); every
                # subsequent layer both consumes and produces genuine SO(3) signals, since
                # no smoothing collapses it back down to S2 in between.
                if i == 0:
                    self.convolutions.append(SO3Conv(
                        nfeature_in=self.fs[i]['f_in'],
                        nfeature_out=self.fs[i]['f_out'],
                        L=self.fs[i]['L'],
                    ))
                else:
                    self.convolutions.append(SO3toSO3Conv(
                        nfeature_in=self.fs[i]['f_in'],
                        nfeature_out=self.fs[i]['f_out'],
                        L=self.fs[i]['L'],
                    ))
            else:
                self.convolutions.append(SpinSO3Conv(
                    nfeature_in=self.fs[i]['f_in'],
                    nfeature_out=self.fs[i]['f_out'],
                    L=self.fs[i]['L'],
                ))

            # Everything after the convolution is skipped on the last layer in
            # `forward`, so building it there would only create dead parameters.
            if i == self.n_layers - 1:
                continue

            if self.layer_style in ["so3", "full_so3"]:
                self.iFFTs.append(SO3iFFT(
                    L= self.fs[i]['L'],
                    n_alpha = self.fs[i]['a'],
                    n_beta = self.n_beta,
                    n_gamma = self.fs[i]['a'],
                ))
            else:
                self.iFFTs.append(S2iFFT(
                    L= self.fs[i]['L'],
                    n_alpha = self.fs[i]['a'],
                    n_beta = self.n_beta,
                    n = self.fs[i]['out_spins'],
                ))

            if self.layer_style in ["so3", "full_so3"]:
                if self.use_batch_norm:
                    self.batchnorm.append(
                        FullSpectralBatchNormalization(spins = self.fs[i]['out_spins'],
                                                    nfeatures = self.fs[i]['f_out']))
                if self.use_layer_norm:
                    self.layernorm.append(
                        FullSpectralLayerNormalization(n = self.fs[i]['out_spins'], nfeatures = self.fs[i]['f_out']))
                if self.nonlinearity_type == "relu":
                    self.nonlinearities.append(WeightedReLU())
                elif self.nonlinearity_type == "leaky_relu":
                    self.nonlinearities.append(WeightedLeakyReLU())
                # "full_so3" never smooths between layers -- only "so3" does, every layer.
                if self.layer_style == "so3":
                    self.spatial_smoothing.append(SpatialSmoothing(n=self.fs[i]['out_spins']))
            else:
                if self.use_batch_norm:
                    self.batchnorm.append(
                        SpinSpectralBatchNormalization(nfeatures = self.fs[i]['f_out'],
                                                    n = self.fs[i]['out_spins']))
                if self.use_layer_norm:
                    self.layernorm.append(
                        SpinSpectralLayerNormalization(nfeatures = self.fs[i]['f_out'],
                                                    n = self.fs[i]['out_spins']))
                if self.nonlinearity_type == "phase_collapse":
                    self.nonlinearities.append(PhaseCollapseNonlinearity(
                                                    n=self.fs[i]['out_spins'],
                                                    nfeatures=self.fs[i]['f_out']))
                elif self.nonlinearity_type == "magnitude":
                    self.nonlinearities.append(
                        MagnitudeNonlinearity(n=self.fs[i]['out_spins']))

            self.dropouts.append(ComplexDropout(p=0.2))

            if self.layer_style == "full_so3":
                self.FFTs.append(SO3FFT(L=self.fs[i]['L']))
            else:
                self.FFTs.append(S2FFT(
                    L=self.fs[i]['L'],
                    n=self.fs[i]['out_spins'],
                ))

        # Input and output transforms
        self.s2fft = S2FFT(L=L, n=self.spins[0])
        self.s2ifft = S2iFFT(L=L, n_alpha=self.n_alpha, n_beta=self.n_beta, n=self.spins[-1])

        # Only the last layer needs to convert from SO(3) to S2
        if self.layer_style in ["so3", "full_so3"]:
            self.so3_to_s2 = SO3_to_S2(L=L, n=self.spins[-1])

    def get_buffered_slice_wigner(self, n=1):
        """
        Return the sliced Wigner D matrix list on the current device.
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
        """
        return self.get_buffer("wigner_full")

    def forward(self, x):
        batch_size = x.size(0)

        # If the input is a scalar, it might be encoded as a float32 tensor.
        # We convert it to a complex tensor to be able to use the SO3 convolutions.
        if x.dtype == torch.float32:
            x = torch.view_as_complex(x)
        
        # If the input has only one feature, the feature dimension might be missing.
        if len(x.shape) < 4:
            x = x.reshape(batch_size, 1, self.n_alpha, self.n_beta)

        x = self.s2fft(x,
                       d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                       d_matrices_n1=self.get_buffered_slice_wigner(n=1))

        # Sequential layers - all at the same resolution, no pooling/unpooling, no skip connections
        for i in range(self.n_layers):
            x = self.convolutions[i](x)

            # No batch norm / nonlinearity / dropout on the last layer
            if i < self.n_layers - 1:
                if self.use_batch_norm:
                    x = self.batchnorm[i](x)
                if self.use_layer_norm:
                    x = self.layernorm[i](x)

                if self.layer_style in ["so3", "full_so3"]:
                    x = self.iFFTs[i](x, d_matrices=self.get_buffered_full_wigner())
                else:
                    x = self.iFFTs[i](x,
                                     d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                                     d_matrices_n1=self.get_buffered_slice_wigner(n=1))

                x = self.nonlinearities[i](x)
                x = self.dropouts[i](x)

                # "full_so3" never smooths between layers -- the signal stays on SO(3)
                if self.layer_style == "so3":
                    x = self.spatial_smoothing[i](x)

                if self.layer_style == "full_so3":
                    x = self.FFTs[i](x, d_matrices=self.get_buffered_full_wigner())
                else:
                    x = self.FFTs[i](x,
                                     d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                                     d_matrices_n1=self.get_buffered_slice_wigner(n=1))

        # If x is in the SO(3) domain, we slice to S2
        if self.layer_style in ["so3", "full_so3"]:
            x = self.so3_to_s2(x)
        
        x = self.s2ifft(x,
                        d_matrices_n0=self.get_buffered_slice_wigner(n=0),
                        d_matrices_n1=self.get_buffered_slice_wigner(n=1))

        x = x.reshape(batch_size, self.n_alpha, self.n_beta)
        
        return x
