import torch
import torch.nn as nn


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
        n: provided if the input x \in X_n (this doesn't really matter for the convolution as of now)
        """
        super(SO3Conv, self).__init__()

        self.nfeature_in = nfeature_in
        self.nfeature_out = nfeature_out
        self.L = L

        self.kernel = torch.ones(self.L*(self.L+2)+1, dtype=torch.complex64).repeat(nfeature_in, nfeature_out,1)

        # Normalized initialization of the filter
        for l in range(0, self.L+1):
            self.kernel[...,(l-1)*(l+1)+1:l*(l+2)+1] *= 1/((2*l+1)*self.nfeature_out)
        self.kernel = nn.parameter.Parameter(self.kernel)

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

        self.kernel = torch.ones(self.L+1, dtype=torch.complex64).repeat(nfeature_in, nfeature_out,1)
        # Normalized initialization of the filter
        for l in range(0, self.L+1):
            self.kernel[:,:,l] *= 1/((2*l+1)*self.nfeature_out)
        self.kernel = nn.parameter.Parameter(self.kernel)
    
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