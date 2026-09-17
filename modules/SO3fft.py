"""Spherical and SO(3) transforms.

Encoding, grids, quadrature and table layout: see `CONVENTIONS.md`.
Algorithm: Kostelec & Rockmore 2008, sec. 3.
"""

import numpy as np
import torch
from .SO3lib import from_index_to_beta

class S2_to_SO3(torch.nn.modules.Module):
    """
    This module converts an S2 signal (represented as an n-equivariant signal on SO(3)) to a general
    (not necessarily equivariant) signal on SO(3) by padding it.

    SPECTRAL -> SPECTRAL
    """
    def __init__(self, L:int, n:int|list[int]) -> None:
        """
        L: order of the signal.
        n: Integer or list of integers. Spin/equivariance-type.
        """
        super(S2_to_SO3, self).__init__()

        if isinstance(n, list):
            for n_i in n:
                if n_i != 0 and n_i != 1:
                    raise Exception(f"Only 0 and 1 are accepted values for n. n={n_i} was passed.")
            if len(n) == 1:
                n = n[0]  # If n is a list with one element, we can use it as an int
        elif n != 0 and n != 1:
            raise Exception(f"Only 0 and 1 are accepted values for n. n={n} was passed.")
        self.n = n
        self.L = L
    
    def _forward(self, s2_x:torch.Tensor, n:int) -> list[torch.Tensor]:
        """
        Vectorized version of the forward function where n is consistent across all features.

        Parameters:
        - s2_x: data on S2 as a tensor of shape [batch, features, l/m]
        - n: type of data (n=0 scalar field, n=1 vector field). Integer

        returns: data on SO(3) (general, not necessarily equivariant). Shape of the output is not
                dependent on n.
        """

        batch_size = s2_x.size(0)
        nfeatures = s2_x.size(1)

        if n==0:
            so3_x = []
            for l in range(0,self.L+1):
                temp = torch.zeros((s2_x.size(0), s2_x.size(1), 2*l+1,2*l+1),
                                   dtype=torch.complex64, device=s2_x.device)
                temp[...,l+0] = s2_x[...,(l)*(l):(l+1)*(l+1)] #Plug in the column n=0
                so3_x.append(temp)
        elif n == 1:
            so3_x = []
            # Append dummy zero coefficient for n=1
            so3_x.append(torch.zeros((s2_x.size(0), s2_x.size(1), 1, 1),
                                     dtype=torch.complex64, device=s2_x.device))
            
            for l in range(1,self.L+1):
                temp = torch.zeros((s2_x.size(0), s2_x.size(1), 2*l+1,2*l+1),
                                   dtype=torch.complex64, device=s2_x.device)
                temp[...,l+1] = s2_x[...,(l-1)*(l+1)+1:(l)*(l+2)+1] #Plug in the column n=1
                so3_x.append(temp)
        return so3_x

    def forward(self, s2_x:torch.Tensor) -> list[torch.Tensor]:
        """
        Function to handle different types of data (n) and call the _forward function accordingly.

        Parameters:
        - s2_x: data on S2 as a tensor of shape [batch, features, l/m]

        returns: data on SO(3) (general, not necessarily equivariant). Shape of the output is not
                dependent on n.
        """
        if isinstance(self.n, list): #If n is a list, we have both n=0 and n=1 features (half/half)
            batch_size = s2_x.size(0)
            nfeatures = s2_x.size(1)
            
            so3_x = []

            so3_x_0 = self._forward(s2_x[:,:nfeatures//2], n=0) #Half of the features are n=0
            so3_x_1 = self._forward(s2_x[:,nfeatures//2:], n=1) #Half of the features are n=1
            for l in range(0,self.L+1): #Concatenate the n=0 and n=1 features
                so3_x.append(torch.cat((so3_x_0[l], so3_x_1[l]), dim=1))
        else:
            so3_x = self._forward(s2_x, n=self.n)

        return so3_x
    
class SO3_to_S2(torch.nn.modules.Module):
    """
    This module converts a general signal on SO(3) given by Fourier coefficients (l,m,n)
    in an n-equivariant signal on SO(3) given by coefficients (l,m) for n=0 or n=1.

    This is equivalent to applying the smoothing operator for a given n.
    """
    def __init__(self, L:int, n:int|list[int]) -> None:
        super(SO3_to_S2, self).__init__()

        if isinstance(n, list):
            for n_i in n:
                if n_i != 0 and n_i != 1:
                    raise Exception(f"Only 0 and 1 are accepted values for n. n={n_i} was passed.")
            if len(n) == 1:
                n = n[0]  # If n is a list with one element, we can use it as an int
        elif n!=0 and n!=1:
            raise Exception(f"Only 0 and 1 are accepted values for n. n={n} was passed.") 
        
        self.n = n
        self.L = L

    def _forward(self, so3_x:list[torch.Tensor], n:int) -> torch.Tensor:
        """
        Non-vectorized version of the forward function where n needs to be specified.

        Parameters:
        - so3_x: data on SO(3) as a list of tensors
        - n: type of data (n=0 scalar field, n=1 vector field). Integer

        returns: data on S2. (Shape of the output is not dependent on n, as we are using the same
                 encoding convention for both n=0 and n=1).
        """
        batch_size = so3_x[0].size(0)
        nfeatures = so3_x[0].size(1)
        
        s2_x = torch.empty((batch_size, nfeatures, (self.L+1)*(self.L+1)),
                           dtype=torch.complex64, device=so3_x[0].device)

        if n == 0:
            for l in range(0, self.L+1):
                s2_x[...,(l-1)*(l+1)+1:(l)*(l+2)+1] = so3_x[l][...,l+0] #Take the column n=0
        elif n == 1:
            s2_x[...,:1] = 0 #Dummy zero coefficient for n=1
            for l in range(1, self.L+1):
                s2_x[...,(l-1)*(l+1)+1:(l)*(l+2)+1] = so3_x[l][...,l+1] #Take the column n=1
        return s2_x
    
    def forward(self, so3_x:list[torch.Tensor]) -> torch.Tensor:
        """
        Function to handle different types of data (n) and call the _forward function accordingly.
        
        Parameters:
        - so3_x: data on SO(3) as a list of [2*l+1, 2*l+1] tensors indexed by l.

        returns: data on S2 as a tensor of shape [batch, features, l/m] (Shape of the output is not
                 dependent on n, as we are using the same encoding convention for both n=0 and n=1).
        """
        if len(so3_x) != self.L+1:
            raise Exception(f"Expected order up to L={self.L} but received L={len(so3_x)-1}")

        if isinstance(self.n, list): #If n is a list, we have both n=0 and n=1 features (half/half)
            nfeatures = so3_x[0].size(1)
            
            s2_x_0 = self._forward([m[:,:nfeatures//2] for m in so3_x], n=0) #Half features are n=0
            s2_x_1 = self._forward([m[:,nfeatures//2:] for m in so3_x], n=1) #Half features are n=1
            s2_x = torch.cat((s2_x_0, s2_x_1), dim=1)
        else:
            s2_x = self._forward(so3_x, self.n)

        return s2_x
    

class S2FFT(torch.nn.modules.Module):
    """Analysis of an n-equivariant signal on SO(3), for n = 0 or n = 1.

    `f(A Z(gamma)) = f(A) e^{-i n gamma}` collapses the gamma integral, leaving
    an FFT over alpha and a sine-weighted quadrature against `d^l_{mn}` over
    beta. Derivation: paper, appendix; conventions: `CONVENTIONS.md`.

    [batch, channels, alpha, beta] -> [batch, channels, (L+1)^2]
    """
    def __init__(self, L, n:int|list) -> None:
        super(S2FFT, self).__init__()
        self.L = L
        if isinstance(n, list):
            for n_i in n:
                if not isinstance(n_i, int):
                    raise Exception(f"n should be integer (or list), but {type(n_i)} was passed.")
                if n_i != 0 and n_i != 1:
                    raise Exception(f"Only 0 and 1 are accepted values for n. n={n_i} was passed.")
            if len(n)==1:
                n = n[0] #If n is a list with one element, we can use it as an int
        elif n!=0 and n!=1:
            raise Exception(f"Only 0 and 1 are accepted values for n. n={n} was passed.") 
        self.n = n

    def _forward(self, f:torch.Tensor, n:int, d_matrices:list[torch.Tensor]=None) -> torch.Tensor:
        """
        Vectorized version of the forward function where n needs to be specified and the same for
        all features.

        Parameters:
        - f: function in spatial coordinates with shape [batch, features, alpha, beta]
        - n: type of data (n=0 scalar field, n=1 vector field). Integer
        - d_matrices: list of small d_matrices for n=0 or n=1, depending on the value of n.

        returns: function in spectral coordinates with shape [batch, features, l/m]
        """
        batch_size = f.size(0)
        n_features = f.size(1)
        n_alpha = f.size(2)
        n_beta = f.size(3)
        
        maf = (n_alpha-1)//2 #Middle After Flipping
        
        f_tilde = torch.fft.fft(f, dim=-2, norm="ortho")# [batch, feature_in, m, beta, complex]
        f_tilde = torch.fft.fftshift(f_tilde, dim=-2)#[:,:,n_alpha//2-L:n_alpha//2+L+1,:]
        f_tilde = torch.flip(f_tilde, dims=(-2,)) #m needs to be taken in negative order later

        angle = from_index_to_beta(torch.arange(0,n_beta,device=f.device), n_beta=n_beta).to(torch.complex64)
        aa = torch.sin(angle)

        # [batch, channels, l+m]
        f_hat = torch.zeros((batch_size, n_features, self.L*(self.L+2)+1),
                            dtype=torch.complex64, device=f.device) 

        if n==0: #Usual transform
            for l in range(0,self.L+1):
                ww = d_matrices[l]
                ff = f_tilde[...,maf-l:maf+l+1,:].to(f.device)
                f_hat[...,(l-1)*(l+1)+1:l*(l+2)+1] = (2*l+1)/2 \
                    * torch.einsum("b,bm,...mb->...m",aa,ww,ff)
        elif n==1: #Suboptimal transform (1 dummy zero-coefficient at the beginning) 
            for l in range(1,self.L+1):
                ww = d_matrices[l-1]
                ff = f_tilde[...,maf-l:maf+l+1,:].to(f.device)
                f_hat[...,(l-1)*(l+1)+1:l*(l+2)+1] = (2*l+1)/2 \
                    * torch.einsum("b,bm,...mb->...m",aa,ww,ff)
        
        f_hat *= torch.pi/(n_beta+1) # Quadrature length,there are n_beta+1 intervals
        
        return f_hat

    def forward(self, f:torch.Tensor, d_matrices_n0, d_matrices_n1) -> torch.Tensor:
        """
        Function to handle different types of data (n) and call the _forward function accordingly.

        Parameters:
        - f: function in spatial coordinates x1, x2 with batch size and multiple features
             shape: [batch, features, x1, x2]
        
        - d_matrices_n0: list of small d-matrices, columns for n=0 (starting at L=0)
        - d_matrices_n1: list of small d-matrices, columns for n=1 (starting at L=1)

        returns: function in spectral coordinates with shape [batch, features, l/m]
        """
        batch_size = f.size(0)
        nfeatures = f.size(1)

        # If n is a list, then we have multiple features with different n
        if type(self.n) == list:
            f_hat = torch.zeros((batch_size, nfeatures, self.L*(self.L+2)+1), 
                                dtype=torch.complex64, device=f.device)

            f_hat [:,:nfeatures//2] = self._forward(f[:,:nfeatures//2],
                                                    n=0,
                                                    d_matrices=d_matrices_n0)
            f_hat [:,nfeatures//2:] = self._forward(f[:,nfeatures//2:],
                                                    n=1,
                                                    d_matrices=d_matrices_n1)
            
        # If n is a number, then we can apply the operation directly
        else:
            if self.n == 0:
                f_hat = self._forward(f,
                                    n=self.n,
                                    d_matrices=d_matrices_n0,
                                    )
            elif self.n == 1:
                f_hat = self._forward(f,
                                    n=self.n,
                                    d_matrices=d_matrices_n1,
                                    )

        return f_hat
    

class S2iFFT(torch.nn.modules.Module):
    """
    This module performs an iFFT on an SO(3) n-equivariant, for either n=0 or n=1.
    """
    def __init__(self, L:int, n:int|list[int]=1, n_alpha:int=360, n_beta:int=179) -> None:
        """
        Parameters:
        - L: maximum order
        - n: type of data (n=0 scalar field, n=1 vector field). Can be a list
        - n_alpha: resolution of the output in the alpha coordinate
        - n_beta: resolution of the output in the beta coordinate

        returns: function in spatial coordinates with shape [batch, features, alpha, beta]
        """
        super(S2iFFT, self).__init__()
        self.L = L
        self.n_alpha = n_alpha
        self.n_beta = n_beta

        #Sanity check for n and casting
        if type(n)==list:
            for n_i in n:
                if n_i!=0 and n_i!=1:
                    raise Exception(f"Only 0 and 1 are accepted values for n. n={n_i} was passed.")
            if len(n)==1:
                n = n[0] #If n is a list with one element, we can use it as an int
        elif n!=0 and n!=1:
            raise Exception(f"Only 0 and 1 are accepted values for n. n={n} was passed.") 
            
        self.n = n

    def _forward(self, f_hat:torch.Tensor, n:int,
                 d_matrices:list[torch.Tensor]=None,
                 ) -> torch.Tensor:
        """
        Parameters:
        - f_hat: function in spectral coordinates l,m,n with batch size and multiple features
            shape: [batch, features, l/m]
        - n: type of data (n=0 scalar field, n=1 vector field). Integer
        - d_matrices: list of sliced d_matrices for n=0 and n=1

        returns: function in spatial coordinates with shape [batch, features, alpha, beta]
        """
        batch_size = f_hat.size(0)
        n_features = f_hat.size(1)
        
        
        # Initialize an empty array to compute \tilde f_m(\beta).
        # The part that does not get filled inside this module is zero padding
        f = torch.zeros((batch_size, n_features, self.n_alpha, self.n_beta),
                        dtype=torch.complex64, device=f_hat.device)
        
        mid = self.n_alpha//2
        
        if n == 0:
            for l in range(0,self.L+1):
                xx = f_hat[...,(l-1)*(l+1)+1:(l)*(l+2)+1]
                ww = d_matrices[l]
                # It is ok for m to be negative, as the signal is periodic
                f[...,mid-l:mid+l+1, :] += torch.einsum("...m,bm->...mb",xx,ww)

        elif n == 1:
            for l in range(1,self.L+1):
                xx = f_hat[...,(l-1)*(l+1)+1:l*(l+2)+1]
                ww = d_matrices[l-1]
                #It is ok for m to be negative, as the signal is periodic
                f[...,mid-l:mid+l+1, :] += torch.einsum("...m,bm->...mb",xx,ww)
        else:
            raise Exception("Not implemented")
        
        f = torch.fft.fftshift(f, dim=-2)
        f = torch.fft.fft(f, dim=-2, norm="ortho") # [batch, feature_in, alpha, beta, complex]

        return f

    def forward(self, f_hat:torch.Tensor,
                d_matrices_n0:list[torch.Tensor]=None, d_matrices_n1:list[torch.Tensor]=None,
                ) -> torch.Tensor:
        """
        Function to handle different types of data (n) and call the _forward function accordingly.

        Parameters:
        - f_hat: function in spectral coordinates [batch, features, l/m]
        - d_matrices_n0: list of small d-matrices, columns for n=0 (starting at L=0)
        - d_matrices_n1: list of small d-matrices, columns for n=1 (starting at L=1)

        returns: function in spatial coordinates with shape [batch, features, alpha, beta]
        """
        if d_matrices_n0 == None and d_matrices_n1 == None:
            raise Exception("At least one between d_matrices_n0 and d_matrices_n1 is needed.") 
        
        batch_size = f_hat.size(0)
        nfeatures = f_hat.size(1)

        # If n is a list, then we have multiple features with different n
        if isinstance(self.n, list):
            f = torch.zeros((batch_size, nfeatures, self.n_alpha, self.n_beta),
                            dtype=torch.complex64, device=f_hat.device)
            f[:,:nfeatures//2] = self._forward(f_hat[:,:nfeatures//2],
                                        n=0,
                                        d_matrices=d_matrices_n0)
            f[:,nfeatures//2:] = self._forward(f_hat[:,nfeatures//2:],
                                        n=1,
                                        d_matrices=d_matrices_n1)
        # If n is a number, all features have the same n and we can apply the operation directly
        else:
            if self.n == 0:
                f = self._forward(f_hat,
                                n=self.n,
                                d_matrices=d_matrices_n0)
            elif self.n == 1:
                f = self._forward(f_hat,
                                n=self.n,
                                d_matrices=d_matrices_n1)
        return f



class SO3FFT(torch.nn.modules.Module):
    """
    This module performs an FFT for a general signal on SO(3).
    """
    def __init__(self, L:int, method:str="full") -> None:
        """
        Parameters:
        - L: the expected order of the output
        - method: "full" or "delta"
        """
        super(SO3FFT, self).__init__()

        self.L = L
        self.method = method

        ALLOWED_METHODS = ["full", "delta"]

        if self.method not in ALLOWED_METHODS:
            raise Exception(f"Method {self.method} not in {ALLOWED_METHODS}")
    
    def _f_forward(self, x, d_matrices) -> list[torch.Tensor]:
        """
        Implementation of the SO3FFT with full d-matrices method

        Parameters:
        - x: SO(3) spatial signal with shape [batch, features, alpha, beta, gamma]
        - d_matrices: list of Wigner d-matrices indexed by l
        """
        batch_size = x.size(0)
        n_features = x.size(1)
        n_alpha = x.size(2)
        n_beta = x.size(3)
        n_gamma = x.size(4)

        maf_a = (n_alpha-1)//2 #Middle After Flipping for Alpha
        maf_g = (n_gamma-1)//2 #Middle After Flipping for Gamma
        
        # Inverse of the SO3iFFT convention: ortho over alpha, 1/n_gamma over gamma.
        x = torch.fft.fft(x, dim=-3, norm="ortho") # [batch, feature_in, m, beta, n]
        x = torch.fft.fft(x, dim=-1, norm="forward")
        x = torch.fft.fftshift(x, dim=(-3,-1))
        x = torch.flip(x, dims=(-3,-1)) # We need -m and -n as indeces
        
        x_hat = []
        
        angle = from_index_to_beta(torch.arange(0,n_beta,device=x.device),
                                   n_beta=n_beta).to(torch.complex64)
        aa = torch.sin(angle).to(torch.complex64)
        
        for l in range(0,self.L+1):

            ww = d_matrices[l]

            x_hat.append(torch.pi/(n_beta+1)* (2*l+1)/2 * 
                        torch.einsum("b,bmn,...mbn->...mn",
                                    aa,
                                    ww,
                                    x[...,maf_a-l:maf_a+l+1,:,maf_g-l:maf_g+l+1],
                                    ))
            
        return x_hat #List of mn matrices indexed by l

    def _d_forward(self, x, bdeltas) -> list[torch.Tensor]:
        """
        Implementation of the SO3FFT with delta method
        TODO: Implement this method from the legacy code.
        """
        pass

    def forward(self, x:torch.Tensor, d_matrices:list[torch.Tensor]=None, bdeltas=None)\
          -> list[torch.Tensor]:
        """
        Parameters:
        - x: SO(3) spatial signal with shape [batch, features, alpha, beta, gamma]
        - d_matrices: list of Wigner d-matrices indexed by l

        returns: SO(3) spectral signal with shape [[batch, features, l+1,l+1] for l in range(0,L+1)]
                 (list of tensors)

        The method is implemented both in the case of full Wigner d-matrices and deltas.
        """
        if d_matrices == None and bdeltas == None:
            raise Exception("At least one between d_matrices and bdeltas needs to be provided.")

        if self.method == "full":
            return self._f_forward(x, d_matrices)
        elif self.method == "delta":
            return self._d_forward(x, bdeltas)
        
    
class SO3iFFT(torch.nn.modules.Module):
    """
    This module performs an iFFT for a general signal on SO(3).
    """
    def __init__(self,
                 L:int, method:str="full",
                 n_alpha:int=360, n_beta:int=179, n_gamma:int=360):
        """
        Parameters:
        - L: maximum order
        - method: "full" or "delta"
        - n_alpha: resolution of the output in the alpha coordinate
        - n_beta: resolution of the output in the beta coordinate
        - n_gamma: resolution of the output in the gamma coordinate
        """
        super(SO3iFFT, self).__init__()

        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_gamma = n_gamma
        self.L = L
        self.method = method

        ALLOWED_METHODS = ["full", "delta"]

        if self.method not in ALLOWED_METHODS:
            raise Exception(f"Method {self.method} not in {ALLOWED_METHODS}")

    def _f_forward(self, x_hat:list[torch.Tensor], d_matrices:list[torch.Tensor]) -> torch.Tensor:
        """
        Implementation of the SO3FFT with full d-matrices method
        Parameters:
        - x_hat: SO(3) spectral signal with shape [[batch, features, l+1,l+1] for l in range(0,L+1)]
        - d_matrices: list of Wigner d-matrices indexed by l

        returns: SO(3) spatial signal with shape [batch, features, alpha, beta, gamma]
        """
        batch_size = x_hat[0].size(0)
        n_features = x_hat[0].size(1)

        mid_a = (self.n_alpha)//2 #Middle for Alpha
        mid_g = (self.n_gamma)//2 #Middle for Gamma
        
        x = torch.zeros((batch_size, n_features, self.n_alpha, self.n_beta, self.n_gamma),
                        dtype=torch.complex64, device=x_hat[0].device)
        
        for l in range(0,self.L+1):
            ww = d_matrices[l]
            xx = x_hat[l]
                
            x[..., mid_a-l:mid_a+l+1, :, mid_g-l:mid_g+l+1] += torch.einsum("xfmn,bmn->xfmbn",xx,ww) 

        x = torch.fft.fftshift(x, dim=(-3,-1))
        # alpha uses the same convention as S2iFFT; gamma is left unnormalised so
        # that SpatialSmoothing's 1/n_gamma average returns the coefficient itself.
        x = torch.fft.fft(x, dim=-3, norm="ortho")
        x = torch.fft.fft(x, dim=-1)

        return x

    def _d_forward(self, x_hat, bdeltas=None):
        """
        Implementation of the SO3FFT with delta method
        TODO: Implement this method from the legacy code.
        """
        pass

    def forward(self, x_hat:list[torch.Tensor], d_matrices:list[torch.Tensor]=None, bdeltas=None)\
          -> torch.Tensor:
        """
        Parameters:
        - x_hat: SO(3) spectral signal with shape [[batch, features, l+1,l+1] for l in range(0,L+1)]
                 (list)
        - d_matrices: list of Wigner d-matrices

        returns: SO(3) spatial signal with shape [batch, features, alpha, beta, gamma]
        """
        if d_matrices == None and bdeltas == None:
            raise Exception("At least one between d_matrices and bdeltas needs to be provided.")

        if self.method == "full":
            return self._f_forward(x_hat, d_matrices)
        elif self.method == "delta":
            return self._d_forward(x_hat, bdeltas)