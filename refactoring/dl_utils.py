import torch
import numpy as np
from pyklip.klip import nan_gaussian_filter
import os.path
from pyklip.klip import meas_contrast
import json
import torch.nn.functional as F
from pyklip.fakes import gaussfit2d
import astropy.io.fits as fits 
import torch.nn as nn
import copy
import torch 
import matplotlib.pyplot as plt 
from pyklip.klip import nan_gaussian_filter


def track_SNR(est_residual_obs, curr_iter, track_locs, track_fluxes,app_size=0.3, mask_name=None, vis_dir_iterations='vis_iters', snr_smooth_sigma=0.,psf_pixel_scale_arcsec=0.062, px_mask_numpy=None, save_plot=True):
    # Now, track the signal loss and SNR of injected planets (and real planets, if applicable)
    
    # Exclude bad pixels from SNR calculation
    if px_mask_numpy is not None:
        px_mask_nanned = np.where(px_mask_numpy == 0., np.nan, px_mask_numpy)
    else:
        px_mask_nanned = 1.

    # For convenience, if we have a MASKLWB, we also have to omit the region along the bar from the SNR
    if 'MASKLWB' in mask_name:
        if px_mask_nanned.shape[-1] == 80:
            # We only have a pre-generated pixel map for the 80x80 case
            package_directory = os.path.dirname(os.path.abspath(__file__))
            mask_along_bar = package_directory+f'/data/common_masks/mask_along_bar.npy'
            mask_along_bar = np.load(mask_along_bar)
            mask_along_bar_nanned = np.where(mask_along_bar == 0., np.nan, mask_along_bar)
            px_mask_nanned+=mask_along_bar_nanned # Combine into a single NAN mask

        else:
            print('We should exclude the region along the bar from SNR calculation, but only 80x80 mask is implemented for now')
            print('Omitting')

    residual = est_residual_obs

    masked_residual = residual * px_mask_nanned
    
    curr_iter_snrs, curr_iter_signal_blob_peak = [], []
    #Check SNRs of known synthetic planets that were injected
    for j, loc in enumerate(track_locs):
        # Every element in track_locs must have this structure: fakes = [{'pos_x_px':10., 'pos_y_px': 0.}]
        xpos, ypos = loc['pos_x_px']*psf_pixel_scale_arcsec, loc['pos_y_px']*psf_pixel_scale_arcsec

        # Extract peak of injection without smoothing, for a good tracker of signal loss
        _, annulus, signal_blob, peak_blob = calc_snr(masked_residual[0], xpos, ypos,app_size,np.sqrt(xpos**2 + ypos**2),return_signal_peak=True)

        # Smooth (optionally) when computing actual SNR
        curr_snr, _, _ = calc_snr(masked_residual[0], xpos, ypos,app_size,np.sqrt(xpos**2 + ypos**2), sigma_kernel=snr_smooth_sigma)

        curr_iter_snrs.append(curr_snr)
        curr_iter_signal_blob_peak.append(peak_blob)
        if save_plot:
            # Create folder to save everything in
            if not os.path.isdir(vis_dir_iterations):
                os.makedirs(vis_dir_iterations, exist_ok=True)

            # Create sub-folder for current iteration saves
            vis_dir_iterations_current = os.path.join(vis_dir_iterations, 'iter'+str(curr_iter))
            if not os.path.isdir(vis_dir_iterations_current):
                os.makedirs(vis_dir_iterations_current, exist_ok=True)
                
            plt.figure(figsize=[15,10])
            plt.subplot(231)
            plt.imshow(residual[0], origin='lower')
            plt.title(f'Sum, peak = {np.nansum(residual[0]):.2f},{np.nanmax(residual[0]):.2f} (orig)')
            plt.subplot(232)
            plt.imshow(masked_residual[0], origin='lower', vmin=-10)
            plt.subplot(234)
            plt.imshow(annulus, origin='lower')
            plt.title(f'Stddev = {np.nanstd(annulus):.2f} (annulus synthetic)')
            plt.subplot(235)
            plt.imshow(signal_blob, origin='lower')
            plt.title(f'Sum, peak = {np.nansum(signal_blob):.2f},{peak_blob:.2f},\n pos x, pos y, flux input ={loc["pos_x_px"]:.2f},{loc["pos_y_px"]:.2f},{track_fluxes[j]:.2f}')
            whole_annulus = np.where(~np.isnan(signal_blob), signal_blob, annulus)
            plt.subplot(236)
            plt.imshow(whole_annulus, origin='lower')
            plt.title(f'Whole annulus. \n Peak, mean, stddev = {np.nanmax(whole_annulus):.2f}, {np.nanmean(whole_annulus):.2f}, {np.nanstd(whole_annulus):.2f}')
            plt.suptitle(f'Curr SNR: {curr_snr:.2f}')
            plt.tight_layout()
            plt.savefig(f'{vis_dir_iterations_current}/vis_measurement_INJECTED_iter{curr_iter}_comp{j}.png')
            plt.close()

    curr_iter_snrs = np.array(curr_iter_snrs) # Array with current SNRs of all sources tracked
    curr_iter_signal_blob_peak = np.array(curr_iter_signal_blob_peak) # Array with current peak of all sources tracked
    signal_losses = curr_iter_signal_blob_peak/track_fluxes # Array with current signal loss, relative to initial flux

    return curr_iter_signal_blob_peak, signal_losses, curr_iter_snrs


def hardcoded_match_flux(psf_model,obs_im,px_mask, wavefront, windsize=30, DEVICE='cuda'):
    with torch.no_grad():
        fluxwindsize = windsize
        maskslice = px_mask[:,(int(psf_model.shape[-1]/2) - int(fluxwindsize/2)):(int(psf_model.shape[-1]/2) + int(fluxwindsize/2)),(int(psf_model.shape[-1]/2) - int(fluxwindsize/2)):(int(psf_model.shape[-1]/2) + int(fluxwindsize/2))]
        tot_flux_pred = (maskslice*psf_model[:,(int(psf_model.shape[-1]/2) - int(fluxwindsize/2)):(int(psf_model.shape[-1]/2) + int(fluxwindsize/2)),(int(psf_model.shape[-1]/2) - int(fluxwindsize/2)):(int(psf_model.shape[-1]/2) + int(fluxwindsize/2))]).sum()
        tot_flux_target = (maskslice*obs_im[:,(int(obs_im.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_im.shape[-1]/2) + int(fluxwindsize/2)),(int(obs_im.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_im.shape[-1]/2) + int(fluxwindsize/2))]).sum()
        flux_mismatch_ratio = tot_flux_target / tot_flux_pred
        wavefront.flux_correction.data *=flux_mismatch_ratio.to(DEVICE)
    return wavefront

def inject_fake_planets(obs,obs_noise_map, wavefront_object, optsys_object,fakes, flux_n_5_sigma, smooth_sigma_for_flux=0., pixelscale=0.062,DEVICE='cuda', **kwargs):
    """
    Utility function to inject fake planets to an observation.
    obs = observation (numpy array)
    smooth_sigma_for_flux = Smoothness used to compute the SNR, to ensure that flux_n_5_sigma of the injected planet is actually N times the 5 sigma detection after smoothing.
    """

    # TODO: Get rid of pixelscale keyword and just extract it from something else (e.g. our optsys_object)

    fluxes = generate_flux_rel_5sigma_fundamental_noise(fakes,obs_noise_map,flux_n_5_sigma,smooth_sigma_px=smooth_sigma_for_flux, pixelscale=pixelscale, **kwargs)
    
    # For simplicity, use the same optical system and such to generate the fakes; can create a new one if you want 
    fake_planets = []
    original_injected_peaks = []
    with torch.no_grad(): # Don't compute gradient during these operations
        planet_wavefront = copy.deepcopy(wavefront_object)
        for j,planet in enumerate(fakes):
            offset_planet = nn.Parameter(torch.FloatTensor([planet['pos_x_px'] * arcsec2rad(pixelscale), planet['pos_y_px'] * arcsec2rad(pixelscale)]))
            offset_planet = offset_planet.to(DEVICE)
            planet_wavefront.angles = nn.Parameter(offset_planet)
            planet_psf = optsys_object(planet_wavefront)
            planet_psf = planet_psf.detach().cpu().numpy() # turn to Numpy array for some operations
            smoothed_injected = nan_gaussian_filter(np.squeeze(planet_psf), smooth_sigma_for_flux) # smooth injection with sigma=1 pixel
            
            normaltosmooth = np.nanmax(planet_psf) / np.nanmax(smoothed_injected)
            # smoothed_conversiontorequiredflux = np.nanmax(smoothed_injected) * fluxes[j] # Dividing the smoothed planet by this number gives the correct flux
            curr_pred_planet_scaled = planet_psf / np.nanmax(planet_psf) * fluxes[j] * normaltosmooth# So we divide the pre-smoothed planet by the same flux conversion factor
            curr_pred_planet_scaled = curr_pred_planet_scaled[None] # TODO: check if I need this for the right shape
            fake_planets.append(curr_pred_planet_scaled)
            original_injected_peaks.append(np.nanmax(curr_pred_planet_scaled)) # save peak flux of actual injected planet
            
    fake_planets = np.squeeze(np.array(fake_planets)) # check if I need this for the shape

    if fake_planets.ndim > 2:
        fake_planets_all = np.nansum(fake_planets, axis=0) # If injected many planets in the same image, combine them into a single image to add to the observation
    else:
        fake_planets_all = fake_planets
    
    fake_planets_all = fake_planets_all[None]

    # Inject
    obs+=fake_planets_all

    return obs, original_injected_peaks



def load_and_crop_data(filepath, num_det_px=80, coron_type='MASK335R', crop_params=None):
    """
    Utility function to load JWST data as FITS files and format them for FLIP.

    Current automatic options are 'MASK335R' and 'MASKLWB_NARROW'.
    """
    obs = fits.open(filepath)
    obs_image = obs[1].data
    obs_noise = obs[2].data

    coadded_image = np.mean(obs_image, axis=0)
    coadded_image = np.squeeze(coadded_image) # I forget what the output shape is, so ensure a 2D array

    coadded_noise = np.sqrt(np.sum(obs_noise**2, axis=0)) / len(obs_noise)
    coadded_noise = np.squeeze(coadded_noise)

    if coron_type == 'MASK335R':
        # Assume 320x320 SUBARRAY
        edge1y, edge2y = 160 - int(num_det_px/2), 160 + int(num_det_px/2)
        edge1x, edge2x = 160 - int(num_det_px/2), 160 + int(num_det_px/2)
        shift_y,shift_x = -14, 10
    elif coron_type == 'MASKLWBAR_NARROW':
        # Assume 400x256 SUBARRAY and the NARROW fiducial point override
        edge1y, edge2y = 128 - int(num_det_px/2), 128 + int(num_det_px/2)
        edge1x, edge2x = 200 - int(num_det_px/2), 200 + int(num_det_px/2)
        shift_y,shift_x = -17, -64
    elif crop_params is not None:
        # Apply custom crop to the field of view; must be of the form (example for MASKLWB_NARROW):
        # crop_params = {'hsize':400, 'vsize':256, 'shift_y': -17, 'shift_x': -64}
        edge1y, edge2y = int(crop_params['vsize']/2) - int(num_det_px/2), int(crop_params['vsize']/2) + int(num_det_px/2)
        edge1x, edge2x = int(crop_params['hsize']/2) - int(num_det_px/2), int(crop_params['hsize']/2) + int(num_det_px/2)
        shift_y,shift_x = crop_params['shift_y'], crop_params['shift_x']
    else:
        raise ValueError('You must specify how to crop the observation')

    # NOW WE DO NOT HAVE AN IMAGE CENTERED AT THE ARRAY, so use known coronagraph position to crop at the right pixels
    real_im = coadded_image[(edge1y-shift_y):(edge2y-shift_y), (edge1x-shift_x):(edge2x-shift_x)]
    real_im = real_im.astype(np.float32)

    real_noise = coadded_noise[(edge1y-shift_y):(edge2y-shift_y), (edge1x-shift_x):(edge2x-shift_x)]
    real_noise = real_noise.astype(np.float32)

    # They are guaranteed to be 2D now, but we want 3D tensors, so add a dimension. TODO: clean up
    
    return real_im[None], real_noise[None]


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

def load_model_data(data_dir):
    basic_masks_dir = os.path.join(data_dir, 'pupil_masks')

    aperture = np.load(f'{basic_masks_dir}/primary_transmission.npy')
    aperture = np.flip(aperture, axis=0)

    sampledWFEs = np.load(f'{data_dir}/opds/observation_opd.npy')
    sampledWFEs = np.flip(sampledWFEs, axis=0)[None]

    lyot = np.load(f'{basic_masks_dir}/lyotstop_transmission.npy')

    # WAVELENGTH DEPENDENCE
    data_dir_wldep = os.path.join(data_dir, f'wavelength_dependents')

    wlens_weights = np.load(os.path.join(data_dir_wldep, 'wlens_weights', 'lambda_weights.npy'))
    fpm, nircam_opd = [], []
    for i in range(len(wlens_weights[0])):
        currfpm = np.load(f'{data_dir_wldep}/image_mask_transmissions/image_mask_transmission_{i}.npy')
        curr_nircamopd = np.load(f'{data_dir_wldep}/fov_wl_nircam_opds/fov_wl_nircam_opd_{i}.npy')
        fpm.append(currfpm)
        nircam_opd.append(curr_nircamopd)

    fpm = np.array(fpm)
    nircam_opd = np.array(nircam_opd)

    return aperture, lyot, fpm, nircam_opd, wlens_weights, sampledWFEs


def generate_flux_rel_5sigma_fundamental_noise(fakes,noise_map,nsigma, mask_name='MASK335R', pixelscale=0.062, smooth_sigma_px=0.,annulus_size=4):
    """
    Compute what the flux of the planet should be, assuming we want it to be N times the 5 sigma fundamental noise limit at its separation.
    """
    
    # In pixels relative to center of the image
    positions = [[planet['pos_x_px'], planet['pos_y_px']] for planet in fakes]

    npixels = noise_map.shape[-1]

    noise_realization = np.random.normal(loc=0.0, scale=noise_map, size=noise_map.shape)
    noise_realization = np.squeeze(noise_realization)
    noise_curve_seps, noise_curve_fluxes = meas_contrast(noise_realization, 0, int(npixels/2), annulus_size, low_pass_filter=smooth_sigma_px)
    if mask_name is not None:
        package_directory = os.path.dirname(os.path.abspath(__file__))
        mask_transmission_csv = package_directory+f'/data/coron_transmission_csvs/{mask_name}.csv'
        try:
            instrument_throughput = np.genfromtxt(mask_transmission_csv, delimiter=',')
        except FileNotFoundError:
            raise FileNotFoundError('Valid mask names are MASK210R, MASK335R, MASK430R, MASKLWB_F250M, MASKLWB_F480M, MASKLWB_NARROW, MASKLWB_F182M, MASKLWB_F212N')
    else:
        # assume no mask is used, so uniform throughput = 1
        seps = np.linspace(0,5,100)
        thr = np.ones_like(seps)
        instrument_throughput = np.vstack([seps,thr]).T

    # noise_curve_seps is always non-negative
    contrast_new_test_weighing = np.interp(noise_curve_seps * pixelscale,instrument_throughput[:,0], instrument_throughput[:,1],left=0., right=1.)
    noise_curve_fluxes_wthroughput = noise_curve_fluxes / contrast_new_test_weighing

    fluxes = []

    for position in positions:
        fivesigma_flux = np.interp(np.sqrt(position[0]**2 + position[1]**2),noise_curve_seps, noise_curve_fluxes_wthroughput)
        #onesigma_flux = fivesigma_flux / 5. # Not quite accurate since N sigma significance is not linear with flux, 
        # So instead, we just stipulate now that N is the factor relative to the 5 sigma fundamental noise
        # e.g., N=1 is a 5 sigma detection, N=2 is twice the brightness of a 5 sigma detection, but not necessarily a 10 sigma detection.
        # If we want something linear, use SNR instead (i.e., not using the small number statistics correction from Mawet 2014).

        fluxes.append(fivesigma_flux*nsigma)
    
    return fluxes

def generate_positions_sigma(N, nsigma,noise_map, npixels, inj_pos=None, uniform='cartesian',data_dir='datadir'):
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


def calc_snr(im, offset_x,offset_y, rad_blob, rad_ann, width=0.2, blur_before_signal=False, blur_annulus=True, sigma_kernel=0.0,return_signal_peak=False):
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