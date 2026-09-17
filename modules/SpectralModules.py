import torch
import torch.nn as nn

from .SO3lib import from_index_to_beta, process_n, compute_spins

class WeightedReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones((1,),dtype=torch.complex64))
        self.b = nn.Parameter(torch.zeros((1,),dtype=torch.complex64))

    def forward(self, z):
        z = self.a*z + self.b
        return torch.view_as_complex(torch.relu(torch.view_as_real(z)))
    
class WeightedLeakyReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)

        # Weights
        self.a = nn.Parameter(torch.ones((1,),dtype=torch.complex64))
        self.b = nn.Parameter(torch.zeros((1,),dtype=torch.complex64))
    
    def forward(self, z):
        z = self.a*z + self.b
        return torch.view_as_complex(self.lrelu(torch.view_as_real(z)))
    
class WeightedTanh(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tanh = nn.Tanh()
        self.a = nn.Parameter(torch.ones((1,),dtype=torch.complex64))
        self.b = nn.Parameter(torch.zeros((1,),dtype=torch.complex64))

    def forward(self, z):
        z = self.a*z + self.b
        return torch.view_as_complex(self.tanh(torch.view_as_real(z)))

class RealNonlinearity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        # `view_as_real` aliases the caller's storage, so zeroing through it
        # mutated the input.
        return torch.complex(z.real, torch.zeros_like(z.real))

class ComplexDropout(nn.Module):
    def __init__(self, p) -> None:
        super().__init__()
        self.module = nn.Dropout(p=p)

    def forward(self, z):
        if self.training:
            return torch.view_as_complex(self.module(torch.view_as_real(z)))
        else:
            return z

class SpectralDropout(nn.Module):
    def __init__(self, p) -> None:
        super().__init__()
        self.module = nn.Dropout(p=p)

    def forward(self, z):
        # Assigning into `z` rebound the caller's list.
        return [torch.view_as_complex(self.module(torch.view_as_real(z_l)))
                for z_l in z]

class MagnitudeNonlinearity(nn.Module):
    def __init__(self, n:int|list, spin_0_relu=True) -> None:
        super().__init__()
        self.spin_0_relu = spin_0_relu
        self.spin0, self.spin1 = compute_spins(process_n(n))

        if self.spin0 and self.spin_0_relu:
            self.relu = WeightedLeakyReLU()

        self.EPS = 1e-6
        # `b` shifts the magnitude, which only happens on the spin-1 branch.
        self.uses_b = self.spin1 or not self.spin_0_relu
        if self.uses_b:
            self.b = nn.Parameter(torch.zeros((1,),dtype=torch.float32))

    def forward(self, z):
        if self.spin_0_relu:
            # Spin-0 should be ReLU
            if self.spin0 and self.spin1:
                # Both spins
                z0 = z[:,:z.shape[1]//2]
                z1 = z[:,z.shape[1]//2:]
                mag1 = torch.abs(z1)
                phase1 = z1/(mag1+self.EPS)
                mag1 = torch.relu(mag1+self.b)
                z = torch.cat((self.relu(z0), mag1*phase1), dim=1)
            elif self.spin0 and not self.spin1:
                # Only spin 0
                z = self.relu(z)
            else:
                # Only spin 1
                z = torch.relu(torch.abs(z)+self.b)*(z/(torch.abs(z)+self.EPS))
        else:
            # Only magnitude nonlinearity
            z = torch.relu(torch.abs(z)+self.b)*(z/(torch.abs(z)+self.EPS))
        return z
    
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.weight_r = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight_i = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
        if bias:
            self.bias_r = nn.Parameter(torch.zeros(out_features))
            self.bias_i = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

    def forward(self, x):
        # x: [batch, resolution, resolution, in_features], complex dtype
        # If x is complex, process real and imag
        if torch.is_complex(x):
            x_r = torch.real(x)
            x_i = torch.imag(x)
        else:
            x_r = x
            x_i = torch.zeros_like(x, device=x.device)

        y_r = torch.matmul(x_r, self.weight_r.t()) - torch.matmul(x_i, self.weight_i.t())
        y_i = torch.matmul(x_r, self.weight_i.t()) + torch.matmul(x_i, self.weight_r.t())

        if self.bias_r is not None:
            y_r = y_r + self.bias_r
            y_i = y_i + self.bias_i

        return torch.complex(y_r, y_i)
    
class PhaseCollapseNonlinearity(nn.Module):
    """
    Nonlinearity that concatenates x to |x|, inspired by Guth et al.
    Applies to spin-weighted functions by concatenating the magnitude of all spins
    to the value at spin == 0, then projecting it back to the input number of channels.

    Args:
        n (int): Spin.
        nfeatures (int): Number of input features (channels).
    """
    def __init__(self, n, nfeatures):
        super().__init__()
        self.n = n
        self.nfeatures = nfeatures
        # Two dense layers for spin-0 and |x| to the new spin-0 slice
        ##self.dense_spin0 = nn.Linear(nfeatures//2, nfeatures//2, dtype=torch.complex64)
        ##self.dense_abs = nn.Linear(nfeatures, nfeatures//2, dtype=torch.complex64)

        self.spin0, self.spin1 = compute_spins(process_n(n))
        if self.spin0 and self.spin1:
            self.dense_spin0 = ComplexLinear(nfeatures//2, nfeatures//2)
            self.dense_abs = ComplexLinear(nfeatures, nfeatures//2)
        elif self.spin0:
            # If spin-1 is not present, fallback on a general nonlinearity
            self.nonlinearity = WeightedLeakyReLU()
        elif self.spin1:
            # If only spin-1 are present, the only activation function that does not break equiv.
            # is magnitude-dependant
            self.nonlinearity = MagnitudeNonlinearity(n=self.n)

    def forward(self, x):
        """
        Concatenates the magnitude of all spins to the spin-0 slice and projects it back.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, features, resolution, resolution], complex64.

        Returns:
            torch.Tensor: Output tensor with updated spin-0 slice.
        """
        # x: [batch, resolution, resolution, n_spins, num_channels], complex64
        
        #Check if both spins are present, otherwise fallback either to a magnitude nonlinearity (only spin-1)
        #or to a general nonlinearity (only spin-0)
        if self.spin0 and self.spin1:
            abs_x = torch.abs(x)  # We would like to use the magnitude, but it is not differentiable
            # Select spin-0 slice: [batch, num_channels, resolution, resolution]
            x_spin0 = x[:, :self.nfeatures//2]

            # Permute to [batch, alpha, beta, channels] to use the nn.Linear
            x_spin0 = x_spin0.permute(0, 2, 3, 1).contiguous()
            abs_x = abs_x.permute(0, 2, 3, 1).contiguous()

            # Apply dense layers
            out_spin0 = (
                self.dense_spin0(x_spin0) +
                self.dense_abs(abs_x)
            ) * 0.5

            # Permute back
            out_spin0 = out_spin0.permute(0, 3, 1, 2).contiguous()

            # Replace spin-0 slice in x. Cat is necessary to avoid non-differentiability issues
            x = torch.cat([out_spin0, x[:, self.nfeatures//2:]], dim=1)
        else:
            x = self.nonlinearity(x)
        return x

class SphericalBatchNormalization(nn.Module):
    """
    Batch normalization for spherical functions.

    Two main changes with respect to the usual nn.BatchNorm:
      1) Subtracting/adding a complex value is not rotation-equivariant for spins other than 0.
         Bias is therefore applied only to spin 0 components.
      2) Mean and variance computation on the sphere must take into account the discretization cell
         areas.

    Args:
        n (list or int): Spins.
        nfeatures (int): Number of input features (channels).
        n_beta (int): Number of latitudes (beta samples).
        use_running_stats (bool, optional): If True, use statistics stored in running stats.
        dimensions (int, optional): 2 for spherical signals, 3 for SO(3) signals.
        momentum (float, optional): Decay rate for the exponential moving average of the batch statistics.
        epsilon (float, optional): A small float added to variance to avoid dividing by zero.
        use_bias (bool, optional): If True, add a complex-valued learned bias.
        use_scale (bool, optional): If True, multiply by a complex-valued learned scale.
    """
    def __init__(self, 
                 n: list|int,
                 nfeatures: int,
                 n_beta: int,
                 use_running_stats: bool = True,
                 dimensions: int = 2,
                 momentum: float = 0.01,
                 epsilon: float = 1e-5,
                 use_bias: bool = True,
                 use_scale: bool = True):
        """
        Args:
         n (list or int): Spins.
         nfeatures (int): Number of input features (channels).
         n_beta (int): Number of latitudes (beta samples).
         use_running_stats (bool): If True, use statistics stored in running stats.
         dimensions (int): 2 for spherical signals, 3 for SO(3) signals.
         momentum (float): Decay rate for the exponential moving average of the batch statistics.
         epsilon (float): A small float added to variance to avoid dividing by zero.
         use_bias (bool): If True, add a complex-valued learned bias.
         use_scale (bool): If True, multiply by a complex-valued learned scale.
        """
        super().__init__()
        
        self.nfeatures = nfeatures # Necessary to define the shape of learnable parameters
        self.n_beta = n_beta
        self.use_running_stats = use_running_stats
        self.dimensions = dimensions
        self.momentum = momentum
        self.eps = epsilon
        self.use_bias = use_bias
        self.use_scale = use_scale

        if self.dimensions not in [2, 3]:
            raise Exception("Dimensions must be 2 (spherical) or 3 (SO(3))")

        self.n = process_n(n)
        self.spin0, self.spin1 = compute_spins(self.n)
        
        # Precompute latitude weights (cosine of latitude in radians)
        weights = torch.sin(from_index_to_beta(torch.arange(n_beta), n_beta))  # [lat]
        weights = weights / weights.mean() # normalize for stability

        if self.dimensions == 2:
            self.register_buffer("weights", weights.view(1, 1, 1, -1)) # shape [1,1,1,lat]
        elif self.dimensions == 3:
            self.register_buffer("weights", weights.view(1, 1, 1, -1, 1)) # shape [1,1,1,lat,1]

        #######################################################
        ## Running statistics (running mean only for spin 0) ##
        #######################################################
        if self.use_running_stats:
            # These are saved as buffers, thus not updated by the optimizer. Although, they are persistent, so that they are
            # saved and loaded with the model in the state_dict.
            if self.spin0 and self.spin1:
                # Both spins
                self.register_buffer('running_mean', torch.zeros(nfeatures//2, dtype=torch.complex64))
            elif self.spin0 and not self.spin1:
                # Only spin 0
                self.register_buffer('running_mean', torch.zeros(nfeatures, dtype=torch.complex64))
            else:
                # Only spin 1, no running mean needed
                self.register_buffer('running_mean', None)
            self.register_buffer('running_var', torch.ones(nfeatures, dtype=torch.float32))
        
        #################################################
        ## Learnable parameters (bias only for spin 0) ##
        #################################################
        # Scale can be used with all the spins
        if self.use_scale:
            self.scale = nn.Parameter(torch.ones(nfeatures, dtype=torch.complex64))
        else:
            self.scale = None
        
        # Bias can be used only with spin 0 (if present)
        if self.use_bias:
            if self.spin0 and self.spin1:
                # Both spins
                self.bias = nn.Parameter(torch.zeros(nfeatures//2, dtype=torch.complex64))
            elif self.spin0 and not self.spin1:
                # Only spin 0
                self.bias = nn.Parameter(torch.zeros(nfeatures, dtype=torch.complex64))
            else:
                # Only spin 1, no running mean needed
                self.bias = None
        else:
            self.bias = None

    def _spherical_mean(self, x, weights):
        """
        Compute spherical mean over the spherical grid dimensions.

        Args:
            x (torch.Tensor): Tensor of shape [batch_size, nfeatures, longitude, latitude].
            weights (torch.Tensor): Tensor of shape [1, 1, 1, latitude] with the weights for each latitude.

        Returns:
            torch.Tensor: Spherical mean over the features [n_channels,].
        """
        if self.dimensions == 2:
            return (x*weights).sum(dim=(0, 2, 3))
        elif self.dimensions == 3:
            return (x*weights).sum(dim=(0, 2, 3, 4))
        else:
            raise Exception("Dimensions must be 2 (spherical) or 3 (SO(3))")
    
    def forward(self, x):
        """
        Normalizes the input using batch means and variances.

        Statistics are computed over the batch and spherical dimensions: (0, 1, 2).

        Args:
            x (torch.Tensor): Input tensor of dimensions (batch_size, nfeatures, longitude, latitude).

        Returns:
            torch.Tensor: Normalized inputs (with same shape as inputs).
        """
        ##################################
        # Running stats. comp. and usage #
        ##################################
        # Expect x shape: [N,C,lon,lat] or [N,C,alpha,beta,gamma]
        weights = self.weights.expand_as(x) # [N,1,lon,lat, X] broadcast to [N,C,lon,lat, X]
        # Compute the running statistics

        # If the model is in training mode, compute the batch statistics and update the running stats
        if self.training:
            # Weighted mean and variance over batch+spatial dims
            weighted_sum = self._spherical_mean(x, weights)
            norm_factor = weights.sum() * x.shape[0]
            mean = weighted_sum / norm_factor

            # Here we compute the compute the difference from the mean for both spins, as this is
            # needed for the variance computation. However, we will use only the spin 0 mean
            # to update the running mean and to apply the bias.
            if self.dimensions == 2:
                diff = x - mean.view(1, -1, 1, 1)
            elif self.dimensions == 3:
                diff = x - mean.view(1, -1, 1, 1, 1)

            # E[x^2] vanishes for spin != 0 (paper appendix E); the variance of a
            # complex signal is E[|x|^2], which is also real, as running_var expects.
            weighted_var = self._spherical_mean(diff.real**2 + diff.imag**2,
                                                weights) / norm_factor


            if self.use_running_stats:

                # Update running stats
                if self.spin0 and self.spin1:
                    # Both spins, running mean only for spin 0
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean[:self.nfeatures//2]
                elif self.spin0 and not self.spin1:
                    # Only spin 0
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                else:
                    # Spin 1 only, no running mean update
                    pass

                # Running variance for all spins
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * weighted_var
            
            if self.spin0 and self.spin1:
                # Both spins, use mean only for spin 0
                mean = mean[:self.nfeatures//2]
        
        # If the model is in eval mode, use the running stats registered as buffers (possibly persistent)
        else:
            mean = self.running_mean
            weighted_var = self.running_var

        # Normalize the batch

        if self.dimensions == 2:
            if mean is not None:
                mean = mean.view(1, -1, 1, 1)
            weighted_var = weighted_var.view(1, -1, 1, 1)
        elif self.dimensions == 3:
            if mean is not None:
                mean = mean.view(1, -1, 1, 1, 1)
            weighted_var = weighted_var.view(1, -1, 1, 1, 1)
        
        if self.spin0 and self.spin1:
            # Mean is guaranteed to be not None
            x_hat = torch.cat((x[:,:self.nfeatures//2] - mean, x[:,self.nfeatures//2:]), dim=1)
            x_hat = x_hat / torch.sqrt(weighted_var + self.eps)
        elif self.spin0 and not self.spin1:
            # Mean is guaranteed to be not None
            x_hat = (x - mean) / torch.sqrt(weighted_var + self.eps)
        else:
            # Mean is None
            x_hat = x / torch.sqrt(weighted_var + self.eps)

        ##############################
        # Apply learnable parameters #
        ##############################
        y = x_hat
        if self.bias is not None or self.scale is not None:
            if self.scale is not None:
                if self.dimensions == 2:
                    scale = self.scale.view(1, -1, 1, 1)
                elif self.dimensions == 3:
                    scale = self.scale.view(1, -1, 1, 1, 1)
                y = scale * x_hat
            if self.bias is not None:
                y = y.clone()
                if self.dimensions == 2:
                    bias = self.bias.view(1, -1, 1, 1)
                elif self.dimensions == 3:
                    bias = self.bias.view(1, -1, 1, 1, 1)
                if self.spin0 and self.spin1:
                    # Both spins, bias only for spin 0
                    y[:,:self.nfeatures//2] += bias
                elif self.spin0 and not self.spin1:
                    # Only spin 0
                    y += bias
                else: # Only spin 1, no bias
                    pass

        return y

class FullSpectralBatchNormalization(nn.Module):
    """
    Batch normalization for SO(3)-functions in the spectral domain.
    """
    def __init__(self, spins: list|int, nfeatures: int, use_running_stats: bool = True,
                 momentum: float = 0.01, eps=1e-6):
        super().__init__()

        self.nfeatures = nfeatures
        self.n = process_n(spins)
        self.has_spin0, self.has_spin1 = compute_spins(self.n)
        self.use_running_stats = use_running_stats
        self.momentum = momentum
        self.EPS = eps

        #######################################################
        ## Running statistics (running mean only for spin 0) ##
        #######################################################
        if self.use_running_stats:
            # These are saved as buffers, thus not updated by the optimizer. Although, they are persistent, so that they are
            # saved and loaded with the model in the state_dict.
            if self.has_spin0 and self.has_spin1:
                # Both spins
                self.register_buffer('running_mean', torch.zeros(nfeatures//2, dtype=torch.complex64))
            elif self.has_spin0 and not self.has_spin1:
                # Only spin 0
                self.register_buffer('running_mean', torch.zeros(nfeatures, dtype=torch.complex64))
            else:
                # Only spin 1, no running mean needed
                self.register_buffer('running_mean', None)
            self.register_buffer('running_var', torch.ones(nfeatures, dtype=torch.float32))
    
    def forward(self, x):
        """
        Normalizes the input in the spectral domain.

        Args:
            x (list of torch.Tensor): List of tensors for each order.
        Returns:
            list of torch.Tensor: Normalized tensors.
        """

        # Compute batch statistics if training, otherwise use running stats
        if self.training:
            # The mean is the coefficient x^0_{00} for spin 0 features, and 0 for spin 1
            # features. x[0] is [batch, features, 1, 1], so reduce over the batch only:
            # running_mean is per channel.
            if self.has_spin0 and self.has_spin1:
                batch_mean = torch.mean(x[0][:, :self.nfeatures//2, 0, 0], dim=0)
            elif self.has_spin0:
                batch_mean = torch.mean(x[0][:, :, 0, 0], dim=0)
            else:
                batch_mean = None

            # Variance is the norm over non-zero coefficients
            batch_var = torch.zeros((x[0].shape[0], x[0].shape[1]), dtype=torch.float32, device=x[0].device)
            for l in range(1,len(x)):
                batch_var += torch.sum(torch.abs(x[l])**2, dim=(2,3))/(2*l+1)
            batch_var = torch.mean(batch_var, dim=0)


            if self.use_running_stats:
                # Update running stats
                if batch_mean is not None:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean

                # Running variance for all spins
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Normalize
        if self.has_spin0 and self.has_spin1:
            # Both spins, use mean only for spin 0
            y = []
            for l, x_l in enumerate(x):
                if l==0:
                    x_l_hat = torch.cat((x_l[:,:self.nfeatures//2] - batch_mean.view(1, -1, 1, 1), x_l[:,self.nfeatures//2:]), dim=1)
                else:
                    x_l_hat = x_l
                # Avoid in-place division on a view that may share storage with autograd graph
                x_l_hat = x_l_hat / torch.sqrt(batch_var + self.EPS).view(1, -1, 1, 1)
                y.append(x_l_hat)
        elif self.has_spin0 and not self.has_spin1:
            # Only spin 0
            y = []
            for l, x_l in enumerate(x):
                if l==0:
                    x_l_hat = x_l - batch_mean.view(1, -1, 1, 1)
                else:
                    x_l_hat = x_l
                x_l_hat = x_l_hat / torch.sqrt(batch_var + self.EPS).view(1, -1, 1, 1)
                y.append(x_l_hat)
        else:
            # Mean is None
            y = []
            for l, x_l in enumerate(x):
                # Out-of-place division to avoid modifying input tensors in-place
                x_l_hat = x_l / torch.sqrt(batch_var + self.EPS).view(1, -1, 1, 1)
                y.append(x_l_hat)
        
        return y

class FullSpectralLayerNormalization(nn.Module):
    """
    Layer normalization for SO(3)-functions in the spectral domain.
    """
    def __init__(self, n: int|list, nfeatures: int, eps=1e-6):
        super().__init__()

        self.nfeatures = nfeatures
        self.EPS = eps
        self.n = process_n(n)
        self.has_spin0, self.has_spin1 = compute_spins(self.n)
    
    def forward(self, x):
        """
        Normalizes the input in the spectral domain.

        Args:
            x (list of torch.Tensor): List of tensors for each order, ordered by l starting from l=0.
                    shape of each tensor: [batch, features, m, n]

        Returns:
            list of torch.Tensor: Normalized tensors.
        """
        # Recall that data is a list ordered by l, and each element has shape [batch, features, m, n]

        # The mean is the coefficient x^0_{00} for spin 0 features, and structurally 0
        # for spin 1 features -- averaging those in would bias it, and subtracting from
        # them is not equivariant.
        if self.has_spin0 and self.has_spin1:
            channels_mean = torch.mean(x[0][:, :self.nfeatures//2, 0, 0], dim=1)
        elif self.has_spin0:
            channels_mean = torch.mean(x[0][:, :, 0, 0], dim=1)
        else:
            channels_mean = None

        # Variance is the norm over non-zero coefficients
        channels_var = torch.zeros((x[0].shape[0],), dtype=torch.float32, device=x[0].device)
        for l in range(1,len(x)):
            channels_var += torch.sum((x[l]*x[l].conj()).real, dim=(1,2,3))/(2*l+1)
        #channels_var = torch.mean(channels_var, dim=1)
        channels_var = channels_var / self.nfeatures

        # Normalize
        y = []
        for l, x_l in enumerate(x):
            y_l = x_l
            if l == 0 and channels_mean is not None:
                mean = channels_mean.view(-1, 1, 1, 1)
                if self.has_spin1:
                    y_l = torch.cat((x_l[:, :self.nfeatures//2] - mean,
                                     x_l[:, self.nfeatures//2:]), dim=1)
                else:
                    y_l = x_l - mean
            y.append(y_l / torch.sqrt(channels_var+self.EPS).view(-1, 1, 1, 1))

        return y

class SpinSpectralBatchNormalization(nn.Module):
    """
    Batch normalization for SO(3)-functions in the spectral domain, when the functions are
    n-equivariant, i.e. spin-weighted.
    """
    def __init__(self, n:int, nfeatures: int, use_running_stats: bool = True, momentum: float = 0.05, eps=1e-6, maxL=179, force_use=False):
        super().__init__()

        self.nfeatures = nfeatures
        self.use_running_stats = use_running_stats
        self.momentum = momentum
        self.EPS = eps

        self.force_use = force_use

        self.n = process_n(n)
        self.has_spin0, self.has_spin1 = compute_spins(self.n)
    
        weights = torch.ones(maxL*(maxL+2), dtype=torch.float32)

        # `weights` indexes the flat spectrum with the l=0 entry dropped, so order l
        # occupies [l^2-1, (l+1)^2-1). Paper appendix E: order l carries 1/(2l+1).
        for l in range(1, maxL+1):
            weights[l*l-1:(l+1)*(l+1)-1] /= 2*l+1

        self.register_buffer("order_scaling", weights) # Registering as buffer so they move to device with the model

        if self.use_running_stats:
            # These are saved as buffers, thus not updated by the optimizer. Although, they are persistent, so that they are
            # saved and loaded with the model in the state_dict.
            if self.has_spin0 and self.has_spin1:
                # Both spins
                self.register_buffer('running_mean', torch.zeros(nfeatures//2, dtype=torch.complex64))
            elif self.has_spin0 and not self.has_spin1:
                # Only spin 0
                self.register_buffer('running_mean', torch.zeros(nfeatures, dtype=torch.complex64))
            else:
                # Only spin 1, no running mean needed
                self.register_buffer('running_mean', None)
            self.register_buffer('running_var', torch.ones(nfeatures, dtype=torch.float32))
    
    def forward(self, x):
        """
        Batch-normalizes the input in the spectral domain according to the running statistics.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, features, l/m]
        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """

        # Compute batch statistics if training, otherwise use running stats
        if self.training or self.force_use:
            # The mean is the coefficient x^0_{00} for spin 0 features, and 0 for spin 1 features
            if self.has_spin0 and self.has_spin1:
                batch_mean = torch.mean(x[:, :self.nfeatures//2, 0], dim=0) # Mean of the 0-spin features
            elif self.has_spin0 and not self.has_spin1:
                batch_mean = torch.mean(x[:,:, 0], dim=0) # Mean of all features (all are 0-spin)
            else:
                batch_mean = None # No mean for spin 1 only features

            # Variance is the norm over non-zero coefficients
            batch_var = torch.mean(torch.einsum("m, bfm -> bf", self.order_scaling[:x.size(2)-1],(x[:,:,1:].conj() * x[:,:,1:]).real),dim=0)


            if self.use_running_stats:
                # Update running stats
                if self.has_spin0:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean

                # Running variance for all spins
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Normalize

        if self.has_spin0 and self.has_spin1:
            # Both spins, use mean only for spin 0
            # Subtract mean from spin-0 features (feature axis is 1)
            x_spin0 = x[:, :self.nfeatures//2, :].clone()
            x_spin0[:, :, 0] = x_spin0[:, :, 0] - batch_mean.view(1, -1)
            x = torch.cat([x_spin0, x[:, self.nfeatures//2:, :]], dim=1)
        elif self.has_spin0 and not self.has_spin1:
            # Only spin 0
            # Subtract mean from the l=0 coefficients (last axis index 0)
            x = x.clone()
            x[:, :, 0] = x[:, :, 0] - batch_mean.view(1, -1)
        else:
            # Mean is None (=0) / No mean subtraction for vector fields
            pass
        x = x / torch.sqrt(batch_var + self.EPS).view(1, -1, 1)

        return x
    
class SpinSpectralLayerNormalization(nn.Module):
    """
    Layer normalization for SO(3)-functions in the spectral domain, when the functions are
    n-equivariant, i.e. spin-weighted.
    """
    def __init__(self, n:int, nfeatures: int, use_running_stats: bool = True, eps=1e-6, maxL=179, force_use=False):
        super().__init__()

        self.nfeatures = nfeatures
        self.use_running_stats = use_running_stats
        self.EPS = eps

        self.force_use = force_use

        self.n = process_n(n)
        self.has_spin0, self.has_spin1 = compute_spins(self.n)
    
        weights = torch.ones(maxL*(maxL+2), dtype=torch.float32)

        # `weights` indexes the flat spectrum with the l=0 entry dropped, so order l
        # occupies [l^2-1, (l+1)^2-1). Paper appendix E: order l carries 1/(2l+1).
        for l in range(1, maxL+1):
            weights[l*l-1:(l+1)*(l+1)-1] /= 2*l+1

        self.register_buffer("order_scaling", weights) # Registering as buffer so they move to device with the model
    
    def forward(self, x):
        """
        Batch-normalizes the input in the spectral domain according to the running statistics.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, features, l/m]
        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        # The mean is the coefficient x^0_{00} for spin 0 features, and 0 for spin 1 features
        if self.has_spin0 and self.has_spin1:
            channels_mean = torch.mean(x[:, :self.nfeatures//2, 0], dim=1) # Mean of the 0-spin features
        elif self.has_spin0 and not self.has_spin1:
            channels_mean = torch.mean(x[:,:, 0], dim=1) # Mean of all features (all are 0-spin)
        else:
            channels_mean = None # No mean for spin 1 only features

        # Variance is the norm over non-zero coefficients
        channels_var = torch.mean(torch.einsum("m, bfm -> bf", self.order_scaling[:x.size(2)-1], (x[:,:,1:].conj() * x[:,:,1:]).real), dim=1)

        # Normalize
        if self.has_spin0 and self.has_spin1:
            # Both spins, use mean only for spin 0
            # Subtract mean from spin-0 features (feature axis is 1)
            x_spin0 = x[:, :self.nfeatures//2, :].clone()
            x_spin0[:, :, 0] = x_spin0[:, :, 0] - channels_mean.view(-1, 1)
            x = torch.cat([x_spin0, x[:, self.nfeatures//2:, :]], dim=1)
        elif self.has_spin0 and not self.has_spin1:
            # Only spin 0
            # Subtract mean from the l=0 coefficients (last axis index 0)
            x = x.clone()
            x[:, :, 0] = x[:, :, 0] - channels_mean.view(-1, 1)
        else:
            # Mean is None (=0) / No mean subtraction for vector fields
            pass
        x = x / torch.sqrt(channels_var + self.EPS).view(-1, 1, 1)

        return x

class SpectralSmoothing(torch.nn.modules.Module):
    """
    Applies the smoothing operator in the spectral domain.

    Args:
        L (int): Maximum order.
        n (int or list, optional): Spin(s). Only 0 and 1 are accepted.
    """
    def __init__(self, L, n:int|list=1):
        super(SpectralSmoothing, self).__init__()

        self.L = L #Not really used if not as a sanity check
        if type(n)==list:
            for n_i in n:
                if n_i!=0 and n_i!=1:
                    raise Exception(f"Only 0 and 1 are accepted values for n. n={n_i} was passed.")
            if len(n)==1:
                n = n[0] #If n is a list with one element, we can use it as an int
        elif n!=0 and n!=1:
            raise Exception(f"Only 0 and 1 are accepted values for n. n={n} was passed.") 
        self.n = n
    
    def forward(self, x):
        """
        Applies spectral smoothing to the input.

        Args:
            x (list of torch.Tensor): List of tensors for each order.

        Returns:
            list of torch.Tensor: Smoothed tensors.
        """
        batch_size = x[0].size(0)
        n_features = x[0].size(1)

        if len(x)!=self.L+1:
            raise Exception(f"Expected order up to L={self.L} but received L={len(x)-1}")
        y = []
        def column(x_l, l, n, features):
            """Column n of block l, or zeros: the l=0 block has no n=1 column."""
            if l == 0 and n == 1:
                return torch.zeros((batch_size, features, 1),
                                   dtype=x_l.dtype, device=x_l.device)
            return x_l[..., l + n]

        if type(self.n) == list: #If n is a list, we have both n=0 and n=1 features (half/half)
            for l in range(0,self.L+1):
                x_l_0 = column(x[l][:,:n_features//2], l, 0, n_features//2)
                x_l_1 = column(x[l][:,n_features//2:], l, 1, n_features//2)
                y.append(torch.cat((x_l_0, x_l_1), dim=1))
        else: # If n is a number, then all features have the same n and we can apply the operation directly
            for l, x_l in enumerate(x):
                y.append(column(x_l, l, self.n, n_features))
        return y
    
class SpatialSmoothing(torch.nn.modules.Module):
    """
    Applies the smoothing operator in the spatial domain.

    For an SO(3) function x the operation is given by:
    S[f](x1, x2) = (pi/2) int_{0}^{2pi} f(x1,x2,x3)*e^{i n x3} dx3

    Args:
        n (int or list, optional): Spin(s). Only 0 and 1 are accepted.
    """
    def __init__(self, n:int|list=1):
        super(SpatialSmoothing, self).__init__()
        if type(n)==list:
            for n_i in n:
                if n_i!=0 and n_i!=1:
                    raise Exception(f"Only 0 and 1 are accepted values for n. n={n_i} was passed.")
            if len(n)==1:
                n = n[0] #If n is a list with one element, we can use it as an int
        elif n!=0 and n!=1:
            raise Exception(f"Only 0 and 1 are accepted values for n. n={n} was passed.") 
        self.n = n #Either an int or a list of ints
    
    def forward(self, x):
        """
        Applies spatial smoothing to the input SO(3) signal.

        Args:
            x (torch.Tensor): SO(3) signal with shape [batch, features, alpha, beta, gamma].

        Returns:
            torch.Tensor: n-equivariant signal with shape [batch, features, alpha, beta].
        """
        batch_size = x.size(0)
        nfeatures = x.size(1)
        n_alpha = x.size(2)
        n_beta = x.size(3)
        n_gamma = x.size(4)

        # If n is a list, then we have both n=0 and n=1 features (half/half)
        if type(self.n)==list:
            y = torch.empty((batch_size, nfeatures, n_alpha, n_beta), dtype=torch.complex64, device=x.device)
            
            #The 0-integral can be optimized, as it is constant
            integr_0 = torch.exp(1j*0*torch.arange(0,n_gamma, device=x.device)*(2*torch.pi)/n_gamma)
            integr_1 = torch.exp(1j*1*torch.arange(0,n_gamma, device=x.device)*(2*torch.pi)/n_gamma)
            y[:,:nfeatures//2] = torch.einsum("g,...g->...", integr_0, x[:,:nfeatures//2])/n_gamma
            y[:,nfeatures//2:] = torch.einsum("g,...g->...", integr_1, x[:,nfeatures//2:])/n_gamma
        
        # If n is a number, then all features have the same n and we can apply the operation directly
        elif self.n == 0 or self.n==1:
            integr = torch.exp(1j*self.n*torch.arange(0,n_gamma, device=x.device)*(2*torch.pi)/n_gamma)
            y = torch.einsum("g,...g->...", integr, x)/n_gamma
        else:
            raise Exception("Provided n should be a number or a list of numbers")

        return y
