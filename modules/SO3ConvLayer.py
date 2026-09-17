import math

import torch
import torch.nn as nn


def complex_glorot(shape, fan_in: int, fan_out: int) -> torch.Tensor:
    """Complex Glorot initialisation: Rayleigh magnitude, uniform phase.

    Trabelsi et al., "Deep Complex Networks" (ICLR 2018), sec. 3.6. The scale
    is chosen so E[|w|^2] = 2/(fan_in + fan_out), which is the Glorot criterion
    carried over to complex weights. Sampling goes through torch so that
    `torch.manual_seed` reproduces it.
    """
    sigma = 1.0 / math.sqrt(fan_in + fan_out)
    u = torch.rand(shape).clamp_min(torch.finfo(torch.float32).tiny)
    magnitude = sigma * torch.sqrt(-2.0 * torch.log(u))
    phase = torch.rand(shape) * (2 * math.pi) - math.pi
    return torch.polar(magnitude, phase)


class SO3Conv(nn.modules.Module):
    """
    SO(3) group convolution.

    If n is provided then a restriction on the weights is applied
    """

    def __init__(self, nfeature_in:int, nfeature_out:int, L):
        """
        nfeature_in: number of input features
        nfeature_out: number of output features
        L: input bandwidth (precision of the input SOFT grid)
        n: provided if the input x in X_n (this doesn't really matter for the convolution as of now)
        """
        super(SO3Conv, self).__init__()

        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.L = L

        # The contraction is over the input channel alone, so that is the fan-in.
        self.kernel = nn.parameter.Parameter(complex_glorot(
            (nfeature_in, nfeature_out, self.L*(self.L+2)+1),
            fan_in=nfeature_in, fan_out=nfeature_out))

    def forward(self, x):
        """
        x:      [batch, nfeature_in,  L*(L+2)+1] with last dimension independent on the spin
        If the spin is 1 then the first coefficient is guaranteed to be zero.
        
        returns: a list of tensors of shape [[batch, nfeature_out,  l+1, l+1] for l in range(0,L+1)]
        """
        batch_size = x.size(0)

        # Sanity check
        if x.size(2) != self.L*(self.L+2)+1:
            raise Exception(f"Input has spectral shape {x.size(2)} but expected {self.L*(self.L+2)+1} with L={self.L}")

        y = []

        # For each order l compute the convolution with the kernel
        for l in range(0, self.L+1):
            y_l = torch.einsum("ion,...im->...omn", 
                                self.kernel[...,(l-1)*(l+1)+1:l*(l+2)+1],
                                x[...,(l-1)*(l+1)+1:l*(l+2)+1])
            y.append(y_l)

        return y

class SO3toSO3Conv(nn.modules.Module):
    """
    SO(3) -> SO(3) group convolution/correlation.

    Unlike SO3Conv, which lifts an S2-valued (vector) spectral signal to an
    SO(3)-valued (matrix) spectral signal, this layer consumes and produces
    genuine SO(3) spectral signals: lists of per-order tensors of shape
    [batch, features, 2l+1, 2l+1]. This lets multiple SO(3) layers be chained
    without collapsing back down to S2 (no SpatialSmoothing/SO3_to_S2) between
    them. Per the compact-group correlation theorem, correlating two SO(3)
    functions amounts to a per-l matrix product contracting the shared index.
    """

    def __init__(self, nfeature_in:int, nfeature_out:int, L):
        """
        nfeature_in: number of input features
        nfeature_out: number of output features
        L: bandwidth (maximum order) of the input/output SO(3) signal
        """
        super(SO3toSO3Conv, self).__init__()

        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.L = L

        # Both the input channel and the input's own row index are contracted,
        # so the fan counts (2l+1) as well and grows with the order.
        self.kernel = nn.ParameterList(
            nn.parameter.Parameter(complex_glorot(
                (nfeature_in, nfeature_out, 2 * l + 1, 2 * l + 1),
                fan_in=nfeature_in * (2 * l + 1),
                fan_out=nfeature_out * (2 * l + 1)))
            for l in range(0, self.L + 1))

    def forward(self, x):
        """
        x:      list of tensors of shape [batch, nfeature_in, 2l+1, 2l+1] for l in range(0,L+1)

        returns: a list of tensors of shape [[batch, nfeature_out, 2l+1, 2l+1] for l in range(0,L+1)]
        """
        if len(x) != self.L + 1:
            raise Exception(f"Input has {len(x)} orders but expected {self.L + 1} with L={self.L}")

        return [torch.einsum("iokp,...imk->...omp", self.kernel[l], x[l]) for l in range(0, self.L + 1)]


class SpinSO3Conv(nn.modules.Module):
    """
    SO(3) group convolution with same weight for all coefficients of the same order l.
    """
    
    def __init__(self, nfeature_in:int, nfeature_out:int, L):
        """
        feature_in: number of input fearures
        nfeature_out: number of output features
        L: input bandwidth (precision of the input SOFT grid)
        """
        super(SpinSO3Conv, self).__init__()

        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.L = L

        # One weight per order; the contraction is over the input channel alone.
        self.kernel = nn.parameter.Parameter(complex_glorot(
            (nfeature_in, nfeature_out, self.L+1),
            fan_in=nfeature_in, fan_out=nfeature_out))
    
    def forward(self, x):
        """
        x:      [batch, nfeature_in,  L*(L+2)+1] with last dimension independent on the spin
        If the spin is 1 then the first coefficient is guaranteed to be zero.
        
        returns: a tensor of shape [batch, nfeature_out,  L*(L+2)+1]
        """
        batch_size = x.size(0)

        # Sanity check
        if x.size(2) != self.L*(self.L+2)+1:
            raise Exception(f"Input has spectral shape {x.size(2)} but expected {self.L*(self.L+2)+1} with L={self.L}")
        
        y = torch.zeros(batch_size, self.nfeature_out, self.L*(self.L+2)+1, dtype=torch.complex64, device=x.device)

        for l in range(0, self.L+1):
            y[..., (l-1)*(l+1)+1:l*(l+2)+1] = torch.einsum("io,...im->...om",
                                                            self.kernel[...,l],
                                                            x[..., (l-1)*(l+1)+1:l*(l+2)+1])
        
        return y