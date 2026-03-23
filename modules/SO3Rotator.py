from xml import dom
import numpy as np
from sympy import beta
import torch
import pickle

from .SO3fft import S2FFT, S2iFFT
from .SO3lib import random_euler_zyz

class SO3Rotator():
    """
    A class to handle rotations in SO(3) using Wigner D-matrices.

    SPACE -> SPACE

    We can rotate signals by using Wigner D-matrices properties, in particular the fact that they
    form a represenation of the group SO(3). In particular D^l(AB) = D^l(A)D^l(B).
    Therefore we can rotate a signal by first transforming it to the spherical harmonics domain,
    then applying the Wigner D-matrix corresponding to the rotation, and finally transforming it
    back to the spatial domain.
    """
    def __init__(self,
                 rotation_type:str = "full",
                 d_matrices_slice:list=None,
                 d_matrices_full:list=None,
                 device:torch.device=torch.device("cpu"),
                 n_alpha:int=360,
                 n_beta:int=179,
                 FFT_L:int=179,
                 n_angles:int=179,
                 n:int=1,
                 domain="spatial"):
        """
        Parameters:
        - rotation_type: "beta" for random beta rotations, "full" for full rotations, "no" for no
                    rotations
        - d_matrices_slice: precomputed Wigner D-matrices for n=0 or n=1
        - d_matrices_full: precomputed Wigner D-matrices for n=0 or n=1
        - device: the device to use for computations
        - FFT_L: the maximum degree of spherical harmonics to use for FFT transformations.
        - n_alpha: the number of alpha values (longitude) to use
        - n_beta: the number of beta values (latitude) to use
        - n_angles: the discretization of the angles for the rotation_grid.
        - n: the value of n (spin, 0 or 1). 0 is scalar fields, 1 is vector fields
        """

        self.n_angles = n_angles
        self.FFT_L = FFT_L
        self.device = device
        self.rotation_type = rotation_type
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n = n
        self.domain = domain

        if self.n != 0 and self.n != 1:
            raise Exception("SO3Rotator only implemented for n=0 and n=1, not for n="+str(n))

        if self.domain == "spatial":
            # The sliced matrices are used for the FFT and iFFT
            if d_matrices_slice == None:
                if self.n == 1:
                    self.d_matrices_slice = [m.to(self.device) for m in \
                        pickle.load(open("precomp/SLICE_WIGNER_N1_"+str(self.n_beta)+"_"+str(self.FFT_L)+".list","rb"))]
                elif self.n == 0:
                    self.d_matrices_slice = [m.to(self.device) for m in \
                        pickle.load(open("precomp/SLICE_WIGNER_N0_"+str(self.n_beta)+"_"+str(self.FFT_L)+".list","rb"))]
            else:
                self.d_matrices_slice = d_matrices_slice
        
        # The full matrices are used for the rotation in the spectral domain (base_rotation)
        if d_matrices_full == None:
            self.d_matrices_full = [m.to(self.device) for m in \
                        pickle.load(open("precomp/FULL_WIGNER_"+str(self.n_angles)+"_"+str(self.FFT_L)+".list","rb"))]
        else:
            self.d_matrices_full = d_matrices_full

        if self.domain == "spatial":
            self.fft = S2FFT(L=self.FFT_L, n=self.n)
            self.ifft = S2iFFT(L=self.FFT_L, n_alpha=self.n_alpha, n_beta=self.n_beta, n=self.n)

    def base_rotation_full(self, x_hat ,beta_index:int=0, positive_beta:bool=True, alpha:float=0, gamma:float=0):
        """
        Apply a rotation around the Y axis of beta_index+1 (integer) degrees.

        Valid only for beta_index between 0 and 179 (included) as these are the datapoints for which
        the Wigner-d are precomputed.

        It relies on the fact that the Wigner D-matrices are both a basis and a representation of
        the group SO(3), so f(A) = f^l_{mn} D^l_{mn}(A) and f(AB) = f^l_{mn} D^l_{mn}(A)D^l_{mn}(B),
        so that if g is the rotatated signal, g(A) = f(AB), then f(AB) = g(A)D^l_{mn}(B).

        Parameters:
        - x_hat: the input signal in the spectral domain [batch_size, channels, l/m]
        - beta_index: the index of the beta angle to use for the rotation (0 to 179 excluded)
        - positive_beta: if True, the rotation is for beta in [0, 178], so positive coeff. used
        - alpha: the angle of rotation around the Z axis in RADIANS
        - gamma: the angle of rotation around the Z axis in RADIANS

        returns:
        - x_hat_rot: the rotated signal in the spectral domain [batch_size, channels, l/m]
        """
        x_hat_rot = torch.empty(x_hat.shape,
                                dtype=torch.complex64, device=self.device)
        
        # The zero-coefficient is just the average of the signal, so we can avoid rotating it
        # D^0_{00} = 1, so it is the same for all rotations

        x_hat_rot[...,0] = x_hat[...,0]
  
        if positive_beta > 0:
            for l in range(1,self.FFT_L+1):
                m = torch.arange(-l, l+1, device=self.device)
                phase_alpha = torch.exp(-1j * m * alpha)
                phase_gamma = torch.exp(-1j * m * gamma)
                x_hat_rot[...,(l-1)*(l+1)+1:(l)*(l+2)+1] = \
                    torch.einsum("...s,ms,m->...m",
                                x_hat[...,(l-1)*(l+1)+1:(l)*(l+2)+1] * phase_gamma,
                                self.d_matrices_full[l][beta_index],
                                phase_alpha,
                                )
        else:
            for l in range(1,self.FFT_L+1):
                m = torch.arange(-l, l+1, device=self.device)
                phase_alpha = torch.exp(-1j * m * alpha)
                phase_gamma = torch.exp(-1j * m * gamma)
                x_hat_rot[...,(l-1)*(l+1)+1:(l)*(l+2)+1] = \
                    torch.einsum("...s,sm,m->...m",
                                x_hat[...,(l-1)*(l+1)+1:(l)*(l+2)+1] * phase_gamma,
                                self.d_matrices_full[l][beta_index],
                                phase_alpha,
                                )


        return x_hat_rot
    
    def base_rotation_beta(self, x_hat, beta_index:int=0, positive_beta:bool=True):
        x_hat_rot = torch.empty(x_hat.shape,
                                dtype=torch.complex64, device=self.device)
        
        # The zero-coefficient is just the average of the signal, so we can avoid rotating it
        # D^0_{00} = 1, so it is the same for all rotations

        x_hat_rot[...,0] = x_hat[...,0]
  
        if positive_beta > 0:
            for l in range(1,self.FFT_L+1):
                x_hat_rot[...,(l-1)*(l+1)+1:(l)*(l+2)+1] = \
                    torch.einsum("...s,ms->...m",
                                x_hat[...,(l-1)*(l+1)+1:(l)*(l+2)+1],
                                self.d_matrices_full[l][beta_index])
        else:
            for l in range(1,self.FFT_L+1):
                x_hat_rot[...,(l-1)*(l+1)+1:(l)*(l+2)+1] = \
                    torch.einsum("...s,sm->...m",
                                x_hat[...,(l-1)*(l+1)+1:(l)*(l+2)+1],
                                self.d_matrices_full[l][beta_index])


        return x_hat_rot

    def base_rotation_zeta(self, x_hat, alpha:float=0, gamma:float=0):
        """
        Apply a rotation around the Z axis of alpha and beta (float) degrees.

        Parameters:
        - x_hat: the input signal in the spectral domain [batch_size, channels, l/m]
        - alpha: the angle of rotation around the Z axis in RADIANS
        - gamma: the angle of rotation around the Z axis in RADIANS
        """

        for l in range(1,self.FFT_L+1):
            m = torch.arange(-l, l+1, device=self.device)
            x_hat[..., (l-1)*(l+1)+1:(l)*(l+2)+1] = \
                torch.einsum("...s,s->...s",
                            x_hat[..., (l-1)*(l+1)+1:(l)*(l+2)+1],
                            torch.exp(-1j*m*(alpha+gamma)),
                            )

        return x_hat

    def __call__(self, x , rotation:tuple=None, return_rotation:bool=False, 
                 beta:int=None, return_beta:bool=False):
        """
        Apply a rotation around the Y axis of beta (integer) degrees. Uses multiple
        `base_rotation_beta` and `base_rotation_no_beta` calls to achieve the desired rotation. 

        Full rotations in SO(3) are saved in radians, while beta rotations are saved in degrees.

        Parameters:
        - x: the input signal to rotate [batch_size, channels, n_alpha, n_beta] (complex) or
             [batch_size, n_alpha, n_beta] (complex)
        - rotation: a tuple (alpha, beta, gamma) with the angles of the rotation in radians.
        - return_rotation: if set True also the rotation is returned (useful in case this is randomly
                          generated here for the source data, for training, and we want to use the same
                          rotation for the target data)
        - beta: the angle of rotation in degrees. If None, it is randomly generated if
                rotation_type is "random"
        - return_beta: if set True also the paramenter is returned (useful in case this is randomly 
                       generated here for the source data, for training, and we want to use the same
                       rotation for the target data)
        """
        if self.domain == "spatial":
            original_dim = x.ndim
            if original_dim == 3: # Data needs to be [batch_size, channels, n_alpha, n_beta]
                x = x[:,None,...]
                
        if rotation is None and beta is None:
            if not return_beta:
                if self.rotation_type == "full":
                    rotation = random_euler_zyz()
                    alpha, beta, gamma = rotation
                    beta = int(np.round(np.degrees(beta))) # Ensure beta is an integer in degrees
                elif self.rotation_type == "beta":
                    alpha = 0
                    beta = np.random.randint(0, 360)
                    gamma = 0
                elif self.rotation_type == "no":
                    alpha = 0
                    beta = 0
                    gamma = 0
                else:
                    raise Exception("If rotation is not provided, beta must be provided or rotation_type must be 'random'")
            else:
                alpha = 0
                beta = np.random.randint(0, 360)
                gamma = 0
        elif beta is not None:
            alpha = 0
            gamma = 0
        elif rotation is not None:
            alpha, beta, gamma = rotation
            beta = int(np.round(np.degrees(beta))) # Ensure beta is an integer in degrees
        
        beta = beta%360 #beta is from 0 to pi, but we also consider the negative angles
        
        if alpha !=0 or beta!=0 or gamma!=0:

            if self.domain == "spatial":
                x_hat = self.fft(x,
                            d_matrices_n0=self.d_matrices_slice, d_matrices_n1=self.d_matrices_slice)
            else:
                x_hat = x

            if beta != 0:
                # We can contract indeces with the D-matrices
                if beta<=179: # Canonical Wigner d-matrix values
                    if alpha != 0 or gamma != 0:
                        x_hat_rot = self.base_rotation_full(x_hat,beta_index=beta-1,
                                                    positive_beta=True, alpha=alpha, gamma=gamma)
                    else:
                        x_hat_rot = self.base_rotation_beta(x_hat,beta_index=beta-1,
                                                            positive_beta=True)
                elif beta == 180: # Special case, we can do two rotations with precomputed matrices
                    if alpha != 0 or gamma != 0:
                        x_hat_rot = self.base_rotation_full(x_hat,beta_index=178,
                                                    positive_beta=True, alpha=alpha, gamma=gamma)
                        x_hat_rot = self.base_rotation_full(x_hat_rot,beta_index=0,
                                                    positive_beta=True, alpha=alpha, gamma=gamma)
                    else:
                        x_hat_rot = self.base_rotation_beta(x_hat,beta_index=178,
                                                            positive_beta=True)
                        x_hat_rot = self.base_rotation_beta(x_hat_rot,beta_index=0,
                                                            positive_beta=True)
                else: # beta in [181, 359], we use the symmetry property of Wigner d-matrices
                    if alpha != 0 or gamma != 0:
                        x_hat_rot = self.base_rotation_full(x_hat,beta_index=360-beta-1,
                                                    positive_beta=False, alpha=alpha, gamma=gamma)
                    else:
                        x_hat_rot = self.base_rotation_beta(x_hat,beta_index=360-beta-1,
                                                            positive_beta=False)
            else:
                # Only multiply by the phase factor
                x_hat_rot = self.base_rotation_zeta(x_hat, alpha=alpha, gamma=gamma)

            if self.domain == "spatial":
                x_rot = self.ifft(x_hat_rot,
                    d_matrices_n0=self.d_matrices_slice, d_matrices_n1=self.d_matrices_slice)
            else:
                x_rot = x_hat_rot
        else:
            x_rot = x

        if self.domain == "spatial" and original_dim == 3:
                x_rot = x_rot.squeeze(1)
        
        if return_rotation:
            return x_rot, rotation
        elif return_beta:
            return x_rot, beta
        else:
            return x_rot