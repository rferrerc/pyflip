import torch
import numpy as np
from pyklip.klip import nan_gaussian_filter
import os.path
from pyklip.klip import meas_contrast
import json
import torch.nn.functional as F
from pyklip.fakes import gaussfit2d

def shift_image_subpixel(image: torch.Tensor, shift_x: float = -0.12, shift_y: float = -0.11) -> torch.Tensor:
    """
    Shift an image by sub-pixel amounts using bilinear interpolation.
    
    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W) or (N, C, H, W)
        shift_x (float): Amount to shift in the x-direction (positive = right)
        shift_y (float): Amount to shift in the y-direction (positive = down)

    Returns:
        torch.Tensor: Shifted image of same shape
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    N, C, H, W = image.shape

    # Create normalized 2D affine matrix for translation
    theta = torch.tensor([[
        [1, 0, -2 * shift_x / W],  # x shift
        [0, 1, -2 * shift_y / H]   # y shift
    ]], dtype=torch.double, device=image.device)

    # Repeat theta for batch
    theta = theta.expand(N, -1, -1)

    # Create grid and sample
    grid = F.affine_grid(theta, size=image.size(), align_corners=False)
    shifted_image = F.grid_sample(image, grid, mode='bicubic', padding_mode='border', align_corners=False)

    return shifted_image.squeeze(0) if shifted_image.size(0) == 1 else shifted_image

def total_variation_loss(img):
    # Compute the total variation loss, for smoothness
    horizontal_diff = img[:, 1:, :] - img[:, :-1, :]
    vertical_diff = img[:, :, 1:] - img[:, :, :-1]
    tv_loss = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
    return tv_loss

def load_mirror_segment_info(data_dir):
    segments_folder = os.path.join(data_dir, 'mirror_segments')

    segments_masks = np.load(os.path.join(segments_folder, 'segment_masks_1024.npy'))

    with open(os.path.join(segments_folder, 'seg_centers_pixels_1024.json'), 'r') as file:
        segment_centers_pixels = json.load(file)
    
    with open(os.path.join(segments_folder, 'segnames_idx.json'), 'r') as file:
        segment_idx = json.load(file)

    return segments_masks, segment_centers_pixels, segment_idx


def weighted_l1_loss(pred, obs, noise, reduction='mean'):
    """
    Computes a noise-weighted L1 loss, assuming Laplace-distributed noise.
    
    Args:
        pred: Predicted tensor of shape (1, H, W)
        obs: Observed tensor of shape (1, H, W)
        noise: Per-pixel noise (stddev), shape (1, H, W)
        reduction: 'mean', 'sum', or 'none'
    """
    weights = 1.0 / (noise + 1e-8)  # avoid division by zero
    loss = weights * (pred - obs).abs()
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def weighted_smooth_l1_loss(pred, obs, noise, beta=1.0, reduction='mean'):
    """
    Computes a noise-weighted Smooth L1 Loss.
    
    Args:
        pred: Predicted tensor of shape (1, H, W)
        obs: Observed tensor of shape (1, H, W)
        noise: Per-pixel noise (stddev), shape (1, H, W)
        beta: Transition point from L1 to L2 in smooth L1 loss
        reduction: 'mean', 'sum', or 'none'
    """
    # Compute per-pixel smooth L1 loss
    diff = pred - obs
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
    
    # Weight by inverse variance (1 / noise^2)
    weights = 1.0 / (noise ** 2 + 1e-8)  # avoid division by zero
    weighted_loss = loss * weights
    
    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:
        return weighted_loss

def load_data(data_dir, num_wl=None, oversample=None):
    basic_masks_dir = os.path.join(data_dir, 'masks_1024')

    aperture = np.load(f'{basic_masks_dir}/primary_transmission_1024.npy')
    aperture = np.flip(aperture, axis=0)
    lyot = np.load(f'{basic_masks_dir}/circlyotstop_transmission_1024.npy')
    if num_wl is not None or oversample is not None:
        data_dir = os.path.join(data_dir, f'npix_{int(1024*oversample)}_num_wl_{int(num_wl)}')
    # else:
        
        # basic_masks_dir = os.path.join(data_dir, f'mask_{int(1024*oversample)}')

        # aperture = np.load(f'{basic_masks_dir}/primary_transmission_1024.npy')
        # aperture = np.flip(aperture, axis=0)
        # lyot = np.load(f'{basic_masks_dir}/circlyotstop_transmission_1024.npy')

    # WAVELENGTH DEPENDENCE

    wlens_weights = np.load(os.path.join(data_dir, 'wlens_weights', 'lambda_weights.npy'))
    fpm, nircam_opd = [], []
    for i in range(len(wlens_weights[0])):
        currfpm = np.load(f'{data_dir}/mask335r_transmissions/mask335r_transmission_{i}.npy')
        curr_nircamopd = np.load(f'{data_dir}/fov_wl_nircam_opds/fov_wl_nircam_opd_{i}.npy')
        fpm.append(currfpm)
        nircam_opd.append(curr_nircamopd)

    fpm = np.array(fpm)
    nircam_opd = np.array(nircam_opd)

    return aperture, lyot, fpm, nircam_opd, wlens_weights

def generate_positions_sigma(N, nsigma,noise_map, npixels,scitargname=None, inj_pos=None, uniform='cartesian',data_dir='datadir'):
    # Noise map must already be "nanned" with the bad pixel map
    # First generate the N samples
    samples = []

    inner_circle_boundary_px = 4.

    if uniform == 'cartesian':
        while len(samples) < N:
            # Number of samples still needed
            needed = N - len(samples)
            
            # Generate more candidate points
            x = np.random.uniform(-npixels/2, npixels/2, needed)
            y = np.random.uniform(-npixels/2, npixels/2, needed)
            
            # Compute distance from origin
            distances = np.sqrt(x**2 + y**2)
            
            # Filter for distances > b
            valid_mask = distances > inner_circle_boundary_px # Only keep samples outside of the inner 4 pixel-radius circle
            valid_samples = list(zip(x[valid_mask], y[valid_mask]))
            
            samples.extend(valid_samples)
    elif uniform == 'polar':
        while len(samples) < N:
            needed = N - len(samples)

            # Sample r and theta uniformly
            r = np.random.uniform(inner_circle_boundary_px, np.sqrt(npixels**2/4 + npixels**2 / 4), needed)
            theta = np.random.uniform(0, 2 * np.pi, needed)

            # Convert to x, y
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Apply bounding box filter
            mask = (np.abs(x) < 40.) & (np.abs(y) < 40.)

            valid_samples = list(zip(r[mask], theta[mask]))
            samples.extend(valid_samples)

    positions = np.array(samples[:N])  # Ensure exactly N samples
    if inj_pos is not None:
        positions = inj_pos
    else:
        positions = np.array([[-12., 12.]])

    noise_realization = np.random.normal(loc=0.0, scale=noise_map, size=noise_map.shape)
    noise_curve_seps, noise_curve_fluxes = meas_contrast(noise_realization, 0, int(npixels/2), 4., low_pass_filter=1)
    instrument_throughput = np.genfromtxt(f'{data_dir}/MASK335R.csv', delimiter=',')

    contrast_new_test_weighing = np.interp(noise_curve_seps * 0.062,instrument_throughput[:,0], instrument_throughput[:,1],left=0., right=1.)
    noise_curve_fluxes_wthroughput = noise_curve_fluxes / contrast_new_test_weighing

    fluxes = []

    for position in positions:
        fivesigma_flux = np.interp(np.sqrt(position[0]**2 + position[1]**2),noise_curve_seps, noise_curve_fluxes_wthroughput)
        onesigma_flux = fivesigma_flux / 5.

        fluxes.append(onesigma_flux*nsigma)
    
    return positions[:,0], positions[:,1], fluxes

def calc_nfringes(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: int,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    focal_shift: float = 0.0,
):
    """
    Calculates the number of fringes in the output plane.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.

    Returns
    -------
    nfringes : Array
        The number of fringes in the output plane.
    """
    # Fringe size
    diameter = npixels_in * pixel_scale_in
    fringe_size = wavelength / diameter

    # Output array size
    output_size = npixels_out * pixel_scale_out
    if focal_length is not None:
        output_size /= focal_length + focal_shift

    # Fringe size and number of fringes
    return output_size / fringe_size



def nd_coords(npixels, pixel_scales = 1.0, offsets=0.0, indexing: str = "xy",):

    if not isinstance(npixels, tuple):
        npixels = (npixels,)
    if not isinstance(pixel_scales, tuple):
        pixel_scales = (pixel_scales,) * len(npixels)
    if not isinstance(offsets, tuple):
        offsets = (offsets,) * len(npixels)

    def pixel_fn(n, offset, scale):
        pix = torch.arange(n) - (n - 1) / 2.0
        pix *= scale
        pix -= offset
        return pix

    lin_pixels = []
    for n, o, p in zip(npixels, offsets, pixel_scales):
      lin_pixels += [pixel_fn(n, o, p)]

    positions = torch.from_numpy(np.array(np.meshgrid(*lin_pixels, indexing=indexing)))
    positions = positions.to(dtype=torch.float64)

    return torch.squeeze(positions)


def transfer_matrix(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    shift: float = 0.0,
    focal_length: float = None,
    focal_shift: float = 0.0,
    inverse: bool = False,
):
    """
    Calculates the transfer matrix for the MFT.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    shift : float = 0.0
        The shift in the center of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.
    inverse: bool = False
        Is this a forward or inverse propagation.

    Returns
    -------
    transfer_matrix : Array
        The transfer matrix for the MFT.
    """
    # Get parameters
    fringe_size = wavelength / (pixel_scale_in * npixels_in)

    # Input coordinates
    scale_in = 1.0 / npixels_in
    in_vec = nd_coords(npixels_in, scale_in, shift * scale_in)

    # Output coordinates
    scale_out = pixel_scale_out / fringe_size
    if focal_length is not None:
        # scale_out /= focal_length
        scale_out /= focal_length + focal_shift
    out_vec = nd_coords(npixels_out, scale_out, shift * scale_out)

    # Generate transfer matrix
    matrix = 2j * np.pi * torch.outer(in_vec, out_vec)
    if inverse:
        matrix *= -1
        
    return torch.exp(matrix)


def dl_MFT(
    phasor,
    wavelength: float,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    shift = [0.0, 0.0],
    pixel: bool = True,
    inverse: bool = False,
):
    """
    Propagates a phasor using a Matrix Fourier Transform (MFT), allowing for output
    pixel scale and a shift to be specified.

    TODO: Add link to Soumer et al. 2007(?), which describes the MFT.

    Parameters
    ----------
    phasor : Array
        The input phasor.
    wavelength : float, meters
        The wavelength of the input phasor.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    shift : Array = np.zeros(2)
        The shift in the center of the output plane.
    pixel : bool = True
        Should the shift be taken in units of pixels, or pixel scale.


    Returns
    -------
    phasor : Array
        The propagated phasor.
    """
    # Get parameters
    npixels_in = phasor.shape[-1]
    if not pixel:
        shift /= pixel_scale_out

    get_tf_mat = lambda s: transfer_matrix(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        s,
        focal_length,
        0.0,
        inverse,
    )
    x_mat = get_tf_mat(shift[0]).to(phasor.device)
    y_mat = get_tf_mat(shift[1]).to(phasor.device)
    phasor = (y_mat.T @ phasor) @ x_mat

    # Normalise
    nfringes = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    phasor *= np.exp(np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out)))

    return phasor


def partial_MFT(
    npixels_in,
    wavelength: float,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    shift = [0.0, 0.0],
    pixel: bool = True,
    inverse: bool = False,
):
    if not pixel:
        shift /= pixel_scale_out

    get_tf_mat = lambda s: transfer_matrix(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        s,
        focal_length,
        0.0,
        inverse,
    )
    x_mat = get_tf_mat(shift[0])
    y_mat = get_tf_mat(shift[1])
    nfringes = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    mult = np.exp(np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out)))

    return x_mat, y_mat, mult



def pixel_coords(npixels: int, diameter: float):
    coords = nd_coords((npixels,) * 2, (diameter / npixels,) * 2)
    #coords = torch.flip(coords, [1])
    return coords

def crop_to(array, npixels: int):
    npixels_in = array.shape[-1]
    start, stop = (npixels_in - npixels) // 2, (npixels_in + npixels) // 2
    return array[..., start:stop, start:stop]


def cart2polar(coordinates):
    x, y = coordinates
    return torch.array([torch.hypot(x, y), torch.arctan2(y, x)])

def circle(coords, radius):
    return torch.where(coords[0]**2 + coords[1]**2 < radius**2, 1.0, 0.0)


def arcsec2rad(values):
    """
    Converts the inputs values from arcseconds to radians.

    Parameters
    ----------
    values : Array, arcseconds
        The input values in units of arcseconds to be converted into radians.

    Returns
    -------
    values : Array, radians
        The input values converted into radians.
    """
    return values * np.pi / (3600 * 180)

def rad2arcsec(values):
    """
    Converts the inputs values from radians to arcseconds.

    Parameters
    ----------
    values : Array, radians
        The input values in units of radians to be converted into arcseconds.

    Returns
    -------
    values : Array, arcseconds
        The input values converted into arcseconds.
    """
    return values * (3600 * 180) / np.pi


def mask_annulus(im, radius,width=0.3):
    size = im.shape[-1]
    x = np.linspace(-int(size/2) * 0.062424185, int(size/2) * 0.062424185, size)
    y = np.linspace(-int(size/2) * 0.062424185, int(size/2)* 0.062424185, size)
    mesh = np.meshgrid(x,y)
    annulus = np.where((mesh[0]**2 + mesh[1]**2 > (radius-width)**2)&(mesh[0]**2 + mesh[1]**2 < (radius+width)**2), im, np.nan)
    return annulus

def circular_mask(im, peak_x,peak_y, rad_to_mask, value_to_mask, mask_inside=True):
    size = im.shape[-1]
    x = np.linspace(-int(size/2) * 0.062424185, int(size/2) * 0.062424185, size)
    y = np.linspace(-int(size/2) * 0.062424185, int(size/2)* 0.062424185, size)
    mesh = np.meshgrid(x,y)
    if mask_inside:
        circlito = np.where(((mesh[0] - peak_x)**2 + (mesh[1]-peak_y)**2 < rad_to_mask**2), value_to_mask, im)
    else:
        circlito = np.where(((mesh[0] - peak_x)**2 + (mesh[1]-peak_y)**2 > rad_to_mask**2), value_to_mask, im)
    return circlito

# def get_peak_in_circle(im, offset_x,offset_y, rad):
#     origin_x, origin_y = im.shape[0]/2, im.shape[1]/2
#     offset_x_pix, offset_y_pix = offset_x/0.063, offset_y/0.063
#     peakpix_x, peakpix_y = origin_x + offset_x_pix, origin_y+offset_y_pix
#     xcoords = np.arange(im.shape[-1])
#     mesh = np.meshgrid(xcoords,xcoords)
#     circlito = np.where(((mesh[0] - peakpix_x)**2 + (mesh[1]-peakpix_y)**2 < rad**2), im, np.nan)
#     return circlito

def calc_snr(im, offset_x,offset_y, rad_blob, rad_ann, width=0.3, blur_before_signal=False, blur_annulus=True, sigma_kernel=0.0,return_signal_peak=False):
    psf_pixel_scale = 0.062424185
    if not blur_annulus:
        positive_blob = circular_mask(im, offset_x,offset_y, rad_blob, np.nan, mask_inside=False)
    elif blur_before_signal:
        im = nan_gaussian_filter(im, sigma_kernel)
        positive_blob = circular_mask(im, offset_x,offset_y, rad_blob, np.nan, mask_inside=False)
    else:
        positive_blob = circular_mask(im, offset_x,offset_y, rad_blob, np.nan, mask_inside=False)
        im = nan_gaussian_filter(im, sigma_kernel) 

    # This function wants the position in pixels
    xgauss,ygauss = im.shape[0]/2-0.5, im.shape[1]/2-0.5 # 0-indexing, true center in between pixels
    xgauss+=offset_x/psf_pixel_scale
    ygauss+=offset_y/psf_pixel_scale
    signal,_,_,_ = gaussfit2d(positive_blob, xgauss, ygauss, searchrad=5, guessfwhm=3, guesspeak=np.nanmax(positive_blob), refinefit=True)

    positive_annulus = mask_annulus(im, rad_ann, width)

    valid_annulus = np.where((~np.isnan(positive_annulus)&(np.isnan(positive_blob))), positive_annulus, np.nan)

    noise = np.nanstd(valid_annulus)
    if return_signal_peak:
        return signal/noise, valid_annulus, positive_blob, signal
    else:
        return signal/noise, valid_annulus, positive_blob

def merge_injected_planets(array):
    total_inject = np.nansum(array, axis=0)
    new_arr = np.zeros_like(array)
    for i in range(len(new_arr)):
        new_arr[i] = total_inject

    return new_arr