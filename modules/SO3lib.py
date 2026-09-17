import torch
import torch.nn.functional as F
import numpy as np
import scipy
import math
from scipy.ndimage import map_coordinates

def from_index_to_beta(index, n_beta=179):
    """
    The data has been cut so that there are 179 (or other amount) indeces for beta:
    the one corresponding to beta=-90° and +90° for both positive and negative latitudes is removed.
    The latitudes run from -89.75° to +89.75°. The angle is returned in radians.
    """
    return torch.pi*(index+1)/(n_beta+1)

###########################################
### Wigner D-matrices related functions ###
###########################################
def produce_sine_grid(device, n_beta=179):
    return torch.sin(from_index_to_beta(torch.arange(0, n_beta, device=device), n_beta=n_beta))

def beta_exponent(beta, mat):
    """
    Function to vectorize the matrix exponential,
    when the matrix is multiplied by a scalar before exponentiation
    
    :param beta: the latitude angle
    :param mat: the iJ_y matrix
    
    :return: exp(beta*iJ_y) [matrix exponential]
    """
    return scipy.linalg.expm(beta * mat) 

def q_n(l,n):
    return math.sqrt((l-n)*(l+n+1))

def wigner_d_matrix(l, beta):
    """
    Returns the Wigner d matrices with rows indexed by m and columns indexed by n
    at position beta in the ZYZ Euler's angle convention
    
    :param l: order
    :param beta: angle
    
    :return: the Wigner d matrix of order l at angle beta
    """
    
    iJy = np.zeros((2*l+1,2*l+1), dtype=np.float64)
    
    x = list(range(0,2*l))
    y = list(range(1,2*l+1))
    
    rep = np.array([q_n(l,-l+i) for i in range(2*l)]).astype(np.float64)
    
    iJy[x,y] = -rep
    iJy[y,x] = rep
    
    iJy *= 1/2
    
    d = scipy.linalg.expm(- beta * iJy) 
    d = torch.from_numpy(d).requires_grad_(False)
    
    return d

def vectorized_wigner_d_matrix(l, betas):
    """
    Returns the Wigner d matrices with rows indexed by m and columns indexed by n
    at position beta in the ZYZ Euler's angle convention
    
    :param l: order
    :param betas: list of angles
    
    :return: the Wigner d matrix of order l at angle beta
    """
    
    iJy = np.zeros((2*l+1,2*l+1), dtype=np.float64)
    
    x = list(range(0,2*l))
    y = list(range(1,2*l+1))
    
    rep = np.array([q_n(l,-l+i) for i in range(2*l)]).astype(np.float64)
    
    iJy[x,y] = -rep
    iJy[y,x] = rep
    
    iJy *= -1/2
    
    d = vec_beta_exponent(betas, iJy) 
    d = torch.from_numpy(d).to(torch.complex64).requires_grad_(False)
    
    return d

def bDelta(l, Delta):
    bDelta = torch.einsum('m,mn->mn',(-1j)**torch.arange(-l,l+1), Delta)
    return bDelta

###########################################
### Rotation matrices related functions ###
###########################################

def random_euler_zyz(radians=True):
    """
    Generate a random rotation on SO(3) represented by ZYZ Euler angles (alpha, beta, gamma),
    uniformly distributed according to the Haar measure.

    Returns
    -------
    alpha, beta, gamma : float
        Euler angles in radians, with
        alpha ∈ [0, 2π),
        beta  ∈ [0, π],
        gamma ∈ [0, 2π).
    """

    #TODO From radians to degrees, and also fix in the SO3rotator

    alpha = np.random.uniform(0, 2*np.pi)
    gamma = np.random.uniform(0, 2*np.pi)
    u = np.random.uniform(-1, 1)   # uniform in [-1,1]
    beta = np.arccos(u)            # ensures correct distribution
    if radians:
        return alpha, beta, gamma
    else:
        return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)

vec_beta_exponent = np.vectorize(beta_exponent, signature="(),(a,b)->(a,b)")

#############################
### Sphere grid functions ###
#############################

_sphere_cache = {}

def _sphere_grid(n_alpha, n_beta):
    key = (n_alpha, n_beta)
    if key not in _sphere_cache:
        u = np.linspace(0, 2*np.pi, n_alpha)
        v = np.linspace(0, np.pi, n_beta, endpoint=False)
        U, V = np.meshgrid(u, v, indexing="ij")

        XX = np.cos(U) * np.sin(V)
        YY = np.sin(U) * np.sin(V)
        ZZ = np.cos(V)

        # Tangent basis vectors (θ, φ directions)
        # e_theta: azimuthal direction (∂r/∂θ)
        e_theta = np.stack([-np.sin(U),
                             np.cos(U),
                             np.zeros_like(U)], axis=-1)
        # e_phi: polar direction (∂r/∂φ)  
        e_phi   = np.stack([np.cos(U) * np.cos(V),
                            np.sin(U) * np.cos(V),
                            -np.sin(V)], axis=-1)

        _sphere_cache[key] = (XX, YY, ZZ, e_theta, e_phi)
    return _sphere_cache[key]

########################################
### 2D-to Sphere Stitching Functions ###
########################################

def stitch_mnist_to_sphere_simple(image, n_alpha=360, n_beta=179, scale_factor=1, alpha_offset=0):
    """
    Stitch a MNIST image to a sphere using simple stitching using the alpha and beta coordinates as rectangular coordinates.
    The image is assumed to be in the range [0, 255] and is converted to a complex64 tensor in the range [0, 1].

    Implementation assumes that the image is in one of the following formats:
     - (B, C, H, W) for batch size B, channels C, height H, and width W;
     - (C, H, W) for channels C, height H, and width W;
     - (H, W) for height H and width W.
    """
    image = image.to(torch.complex64)

    if image.dim() == 4:
        spherical_image = torch.zeros((image.size(0), image.size(1),n_alpha, n_beta), dtype=torch.complex64)
        image = torch.swapaxes(image, 2, 3)
        if scale_factor != 1:
            size = int(image.size(2) * scale_factor)
            image = torch.nn.functional.interpolate(image, size=(size, size), mode='nearest')
        else: 
            size = image.size(2)
        #spherical_image[:,:,alpha_offset:alpha_offset+size, n_beta//2-size//2:n_beta//2+size//2] = image.to(torch.complex64)
    elif image.dim() == 3:
        spherical_image = torch.zeros((image.size(0),n_alpha, n_beta), dtype=torch.complex64)
        image = torch.swapaxes(image, 1, 2)
        if scale_factor!= 1:
            size = int(image.size(1) * scale_factor)
            image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(size, size), mode='nearest').squeeze(0)
        else: 
            size = image.size(1)
        #spherical_image[:, alpha_offset:alpha_offset+size, n_beta//2-size//2:n_beta//2+size//2] = image.to(torch.complex64)
    else:
        spherical_image = torch.zeros((n_alpha, n_beta), dtype=torch.complex64)
        image = torch.swapaxes(image, 0, 1)
        if scale_factor != 1:
            size = int(image.size(0) * scale_factor)
            image = torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0), size=(size,size), mode='nearest').squeeze(0).squeeze(0)
        else: 
            size = image.size(0)

    
    alpha_begin = alpha_offset % n_alpha
    alpha_end = (alpha_offset + size) % n_alpha
    if alpha_end < alpha_begin:
        spherical_image[...,alpha_begin:n_alpha, n_beta//2-size//2:n_beta//2+size//2] = image[:n_alpha-alpha_begin,:]
        spherical_image[...,0:alpha_end, n_beta//2-size//2:n_beta//2+size//2] = image[n_alpha-alpha_begin:,:]
    else:
        spherical_image[...,alpha_begin:alpha_end, n_beta//2-size//2:n_beta//2+size//2] = image


    return spherical_image

def stitch_mnist_to_sphere_proj(image, n_alpha=360, n_beta=179, scale_factor=1, alpha_offset=-100):
    """
    Stitch MNIST images to a sphere using spherical projection stitching (treating the image as an equirectangular map).

    Implementation assumes that the image is in one of the following formats:
     - (B, C, H, W) for batch size B, channels C, height H, and width W;
     - (C, H, W) for channels C, height H, and width W;
     - (H, W) for height H and width W
    """
    image = image.to(torch.complex64)

    original_dim = image.dim()
    if original_dim == 3:
        image = image.unsqueeze(0)
    elif original_dim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    
    H, W = image.shape[-2], image.shape[-1]
    

    alpha = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False)  + alpha_offset/(2*np.pi)# longitude
    beta = np.linspace(0, np.pi, n_beta)                        # latitude

    alpha_grid, beta_grid = np.meshgrid(alpha, beta)

    # Map to image coordinates
    x_img = (alpha_grid / (2 * np.pi)) * (W - 1)
    y_img = (beta_grid / np.pi) * (H - 1)


    # We assume dimension 4 (batch, feature, alpha, beta)
    spherical_image = torch.zeros((image.size(0), image.size(1), n_alpha, n_beta), dtype=torch.complex64)
    #image = torch.swapaxes(image, 2, 3)
    
    for batch_idx in range(image.size(0)):
        for channel_idx in range(image.size(1)):
            # Interpolate
            sphere_values = map_coordinates(image[batch_idx,channel_idx], [y_img, x_img], order=1, mode='wrap')
            spherical_image[batch_idx, channel_idx,:, :] = torch.swapaxes(torch.tensor(sphere_values, dtype=torch.complex64), -1,-2)

    # Return the image in the original dimension
    if original_dim == 3:
        spherical_image = spherical_image.squeeze(0)
    elif original_dim == 2:
        spherical_image = spherical_image.squeeze(0).squeeze(0)

    return spherical_image


def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = torch.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = torch.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample

def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.to(torch.int64)
    iy = ry.to(torch.int64)

    # obtain four sample coordinates
    ix0 = ix
    iy0 = iy
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2

def project_sphere_on_xy_plane(grid, projection_origin):
    ''' returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere '''

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * torch.abs(torch.tensor(xmin)))
    ry = (qy - ymin) / (2 * torch.abs(torch.tensor(ymin)))

    return rx, ry

def stitch_mnist_to_sphere_spherical(image, n_alpha=360, n_beta=179, projection_origin=None, rescale=False, base_value=0):
    if projection_origin is None:
        projection_origin = (0, 0, 2 + 1e-3)

    # Image needs to be torch tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    grid = [np.swapaxes(_sphere_grid(n_alpha, n_beta)[i], 0, 1) for i in range(3)]

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(image, rx, ry)

    # ensure that only south hemisphere gets projected
    sample *= torch.tensor(grid[2] <= 1, dtype=torch.float32)
    sample += base_value * torch.tensor(grid[2] > 1, dtype=torch.float32)

    if rescale:
        # rescale signal to [0,1]
        sample_min = torch.amin(sample, dim=(1, 2), keepdim=True)
        sample_max = torch.amax(sample, dim=(1, 2), keepdim=True)

        sample = (sample - sample_min) / (sample_max - sample_min)

    return torch.flip(torch.swapaxes(sample,-1,-2), dims=(-2,))  # flip to match the original orientation

#########################################
### Sphere plotting related functions ###
#########################################

def plot_scalar_field(x, ax, facecolors=None):
    """
    Plot a scalar field on the sphere surface.
    """
    from matplotlib import cm  # deferred: training jobs are headless

    facecolors = cm.jet if facecolors is None else facecolors

    # convert to numpy safely
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    # handle dtype
    if np.iscomplexobj(x):
        x = np.abs(x)
    elif not np.issubdtype(x.dtype, np.floating):
        raise ValueError(f"Input must be float or complex, got {x.dtype}")

    n_alpha, n_beta = x.shape[-2], x.shape[-1]
    XX, YY, ZZ, _, _ = _sphere_grid(n_alpha, n_beta)

    surf = ax.plot_surface(
        XX, YY, ZZ,
        cstride=1, rstride=1,
        facecolors=facecolors(x),
        vmin=-1, vmax=1
    )
    ax.set_box_aspect([1.0, 1.0, 0.9])
    return ax, surf

def plot_vector_field(z, ax, scale=0.1, color="k", density=5):
    # convert to numpy
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    else:
        z = np.asarray(z)

    if not np.iscomplexobj(z):
        raise ValueError("Input must be complex (z = -v + iu).")

    # unpack spherical components
    v = -np.real(z)        # theta-direction
    u = np.imag(z)       # phi-direction (note the minus sign!)

    n_alpha, n_beta = u.shape
    XX, YY, ZZ, e_theta, e_phi = _sphere_grid(n_alpha, n_beta)

    # 3D Cartesian vector field from spherical components
    VEC = u[..., None] * e_phi + v[..., None] * e_theta
    Ux, Uy, Uz = VEC[...,0], VEC[...,1], VEC[...,2]

    # Downsample grid
    sl = (slice(None,None,density), slice(None,None,density))
    q = ax.quiver(
        XX[sl], YY[sl], ZZ[sl],
        Ux[sl], Uy[sl], Uz[sl],
        length=scale, color=color, normalize=True
    )

    ax.set_box_aspect([1,1,1])
    return ax, q


def plot_vector_field_magnitude(x, ax):
    from matplotlib import cm  # deferred: training jobs are headless

    #Compute the magnitude
    x = np.array(x)
    heatmap = np.abs(x)

    n_alpha = x.shape[-2]
    n_beta = x.shape[-1]

    #Compute the sphere
    u = np.linspace( 0, 2 * np.pi, n_alpha)
    v = np.linspace( 0, np.pi, n_beta, endpoint=False)

    # create the sphere surface
    XX = np.outer( np.cos( u ), np.sin( v ) )
    YY = np.outer( np.sin( u ), np.sin( v ) )
    ZZ = np.outer( np.ones( np.size( u ) ), np.cos( v ) )

    # ~ ax.scatter( *zip( *pointList ), color='#dd00dd' )
    ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.jet( heatmap ), vmin=0, vmax=1)
    ax.set_box_aspect([1.0, 1.0, 0.9])
    return ax

##############################################
### Gradient computation related functions ###
##############################################

def compute_gradients(image):
    """
    Compute the gradients of the image using Sobel filters. Return the two gradients in the x and y
    directions as a tuple.
    """
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # make sure kernels are on same device as image
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)

    original_dim = image.dim()

    # add batch & channel dims: (1,1,H,W)
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)

    if original_dim == 2:
        grad_x = grad_x.squeeze(0).squeeze(0)
        grad_y = grad_y.squeeze(0).squeeze(0)
    elif original_dim == 3:
        grad_x = grad_x.squeeze(0)
        grad_y = grad_y.squeeze(0)

    return grad_x, grad_y


###############################################
### Helper functions in the spectral domain ###
###############################################

def process_n(n):
    if isinstance(n, int):
        return n
    elif isinstance(n, list) or isinstance(n, tuple):
        for i in n:
            if not isinstance(i, int) or i not in [0,1]:
                raise ValueError(f"n must be an 0, 1, or [0,1] (list). Got {n}.")
        if len(n) == 1:
            return n[0]
        else:
            return n

def normalise_spins(spins):
    """Accept `0`, `[0]` or `[0,1]` per layer and always return lists.

    The model constructors document the bare-int form and validate it, then
    test membership with `0 in entry`, so anything but a list used to raise.
    """
    out = []
    for entry in spins:
        entry = [entry] if isinstance(entry, int) else list(entry)
        for n in entry:
            if n not in (0, 1):
                raise ValueError(f"Spins must be 0 or 1, got {n}")
        out.append(entry)
    return out

def compute_spins(n):
    if isinstance(n, int):
        return n==0, n==1
    elif isinstance(n, list) or isinstance(n, tuple):
        spin0 = False
        spin1 = False
        for i in n:
            if i == 0:
                spin0 = True
            elif i == 1:
                spin1 = True
            if spin0 and spin1:
                break
        return spin0, spin1