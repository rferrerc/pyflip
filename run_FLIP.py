import os
import tqdm
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import math

import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from pyklip.klip import nan_gaussian_filter
from dl_utils import arcsec2rad,calc_snr, circular_mask, merge_injected_planets,load_data,generate_positions_sigma,weighted_smooth_l1_loss,weighted_l1_loss,load_mirror_segment_info,total_variation_loss,shift_image_subpixel

from model_classes import PerMirrorZernikes, OPDOffsetModule, LinearInterpOPD,PTT_OPD,ZernikeBasis,ZernikeModel

from propagation_classes import Wavefront,BroadbandWavefront,PointPropagate


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--measurement_file', default='justdata_bothintegrations.npy', type=str)
    parser.add_argument('--reference_file', default='reference_00001.npy', type=str)
    parser.add_argument('--scene_name', default='HIP65426', type=str)
    parser.add_argument('--exp_name', default='default', type=str)
    parser.add_argument('--iters', default=1000, type=int)
    parser.add_argument('--vis_freq', default=300, type=int)
    parser.add_argument('--lr', default=1e-8, type=float)
    parser.add_argument('--star_offset_x', default=0, help='Initial star offset (in pixel)', type=float)
    parser.add_argument('--star_offset_y', default=0, help='Initial star offset (in pixel)', type=float)
    parser.add_argument('--use_ptt', action='store_true')
    parser.add_argument('--use_linear_interp_opd', action='store_true')
    parser.add_argument('--num_wl', default=21, type=int)
    parser.add_argument('--oversample', default=2, type=int)
    parser.add_argument('--fit_everything', action='store_true')
    parser.add_argument('--num_det_px', default=80, type=int)
    parser.add_argument('--enhance_sci_pos_grad', default=None, type=float)
    parser.add_argument('--num_ints', default=1, type=int)
    parser.add_argument('--OPD_loss_weight', default=None, type=float)
    parser.add_argument('--smooth', default=None, type=float)
    parser.add_argument('--use_simulated_data', action='store_true')
    parser.add_argument('--no_median', action='store_false')
    parser.add_argument('--stage_fluxpos_cutoff_iter', default=0, type=int)
    parser.add_argument('--stage_fit_flat_fields_start_iter', default=0, type=int)
    parser.add_argument('--stage_fit_flat_fields_end_iter', default=0, type=int)
    parser.add_argument('--sci_targ_name', default=None, type=str)
    parser.add_argument('--ref_cutoff_iter', default=500, type=int)
    parser.add_argument('--px_mask_file', default=None, type=str)
    parser.add_argument('--ref_which_int', default=0, type=int)
    parser.add_argument('--sci_which_int', default=0, type=int)
    parser.add_argument('--reference_noisemap', default=None, type=str)
    parser.add_argument('--measurement_noisemap', default=None, type=str)
    parser.add_argument('--fit_Lyot', action='store_true')
    parser.add_argument('--fit_flat_field', action='store_false')
    parser.add_argument('--freeze_position', action='store_false')
    parser.add_argument('--sci_lr_weight', default=1., type=float)
    parser.add_argument('--primaryOPD_basis', default=None, type=str)
    parser.add_argument('--other_OPD_meas', default=None, type=str)
    parser.add_argument('--big_lr_factor', default=10., type=float)
    parser.add_argument('--even_bigger_lr_factor', default=1000., type=float)
    parser.add_argument('--freeze_OTE_OPD_sci', action='store_false')
    parser.add_argument('--track_injected_planet', action='store_true')
    parser.add_argument('--blur_annulus_SNR', action='store_false')
    parser.add_argument('--blur_before_signal', action='store_false')
    parser.add_argument('--inject_how_many_random', default=1, type=int)
    parser.add_argument('--inject_how_many_sigma_flux', default=0, type=int)
    parser.add_argument('--weigh_loss_by_noise', action='store_true')
    parser.add_argument('--inject_the_same_in_all_frames', action='store_true')
    parser.add_argument('--inject_all_in_same_frame', action='store_false')
    parser.add_argument('--opd_filename', default=None, type=str)
    parser.add_argument('--forced_stop_iter', default=None, type=int)
    parser.add_argument('--pxartifactsmap_file', default=None, type=str)
    parser.add_argument('--pxartifactsmap_ref_file', default=None, type=str)
    parser.add_argument('--fluxwindsize', default=30, type=int)
    parser.add_argument('--inj_xpos', default=None, type=float)
    parser.add_argument('--inj_ypos', default=None, type=float)

    args = parser.parse_args()

    # Set up physical model for the observation being optimized on: load mask designs, known aberrations, etc.
    DEVICE = 'cuda'
    wf_npix = 1024
    diameter = 6.603464
    psf_npix = args.num_det_px
    psf_pixel_scale = 0.062424185

    if args.oversample is None and args.num_wl is None:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=args.data_dir)
    else:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=args.data_dir, num_wl=args.num_wl, oversample=args.oversample)
        
    aperture = torch.FloatTensor(aperture.copy()).to(DEVICE)[None]

    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    wlen_weights = torch.FloatTensor(wlen_weights).to(DEVICE)

    sampledWFEs = np.load(f'{args.data_dir}/opds_dates/{args.opd_filename}')
    sampledWFEs = np.flip(sampledWFEs, axis=0)[None]
    sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
    sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch = sampledWFEs.contiguous().to(DEVICE)

    if args.other_OPD_meas is not None:
        sampledWFEs_after = np.load(args.other_OPD_meas)
    else:
        sampledWFEs_after = np.load(f'{args.data_dir}/opds_dates/{args.opd_filename}')
    sampledWFEs_after = np.flip(sampledWFEs_after, axis=0)[None]
    sampledWFEs_after = torch.from_numpy(sampledWFEs_after.copy()).float()
    sampledWFEs_after = F.interpolate(sampledWFEs_after[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    wfe_batch_after = sampledWFEs_after.contiguous().to(DEVICE)


    wfe_batch_list = [wfe_batch, wfe_batch_after]
    ############
    # Set up export directories
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}/{args.exp_name}/initial_fit_reference'
    os.makedirs(vis_dir, exist_ok=True)

    # Numerical constants
    contrast_normalization = 0.003716380479003919
    sim_to_real_scaling = 2.1722325193439986 # from comparing the simulation with the peak brightness of the real data
    photons_normalization = 90578.00102527262 * sim_to_real_scaling *1e4 / 2251.24456327477 # mJy/sr
    peak_flux_star  = nn.Parameter(torch.FloatTensor([photons_normalization / contrast_normalization]))

    # Stellar position
    offset_STAR = nn.Parameter(torch.FloatTensor([args.star_offset_x * arcsec2rad(psf_pixel_scale), args.star_offset_y * arcsec2rad(psf_pixel_scale)]))
    
    # Set up the wavefront object: this will be propagated by our optical model
    wavefronts_list1 = BroadbandWavefront(wf_npix, diameter, wlen_weights[0], peak_flux_star, offset_STAR).to(DEVICE)

    # Set up the propagation model parameters
    shift = [0.0, 0.0]
    pixel = True
    focal_length = None
    npixels = wf_npix
    inverse = False
    true_pixel_scale = diameter / npixels
    psf_pixel_scale_arcsec = psf_pixel_scale
    psf_pixel_scale = arcsec2rad(psf_pixel_scale)
    prop_args = (npixels, wlen_weights[0], true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse)

    # Set up the propagation object, using the chosen parametrization
    # The best results have been found using 'grid', that is a simple 1024x1024 array.

    if args.primaryOPD_basis is None:
        OTE_wfe_parametrization = OPDOffsetModule(1024, 1024)
    else:
        if args.primaryOPD_basis == 'grid':
            OTE_wfe_parametrization = OPDOffsetModule(1024, 1024)
        elif args.primaryOPD_basis == 'ptt':
            labelled_transmission,segment_centers, segment_labels = load_mirror_segment_info(data_dir=args.data_dir)
            OTE_wfe_parametrization = PTT_OPD(labelled_transmission,segment_centers, segment_labels,wf_npix).to(DEVICE)
        elif args.primaryOPD_basis == 'interpOPD':
            OTE_wfe_parametrization = LinearInterpOPD(wfe_batch_list[0], wfe_batch_list[1]).to(DEVICE)
        elif args.primaryOPD_basis[:7] == 'zernike':
            numbasis = int(args.primaryOPD_basis[7:])
            basis = ZernikeBasis(1024, 1024, numbasis)
            OTE_wfe_parametrization = ZernikeModel(basis)
        elif args.primaryOPD_basis[:16] == 'permirrorzernike':
            numbasis = int(args.primaryOPD_basis[16:])
            labelled_transmission,segment_centers, segment_labels = load_mirror_segment_info(data_dir=args.data_dir)
            OTE_wfe_parametrization = PerMirrorZernikes(labelled_transmission,segment_centers, segment_labels,wf_npix,num_zernikes=numbasis)
        else:
            OTE_wfe_parametrization = OPDOffsetModule(1024, 1024)

    # Propagator object
    prop_models = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=args.oversample,OTE_wfe_basis=OTE_wfe_parametrization, num_det_px=args.num_det_px).to(DEVICE) for _ in range(args.num_ints)]

    
    # Coronagraph is not at the center of the detector data that JWST produces, so use known coronagraph position to crop at the right pixels
    edge1, edge2 = 160 - int(args.num_det_px/2), 160 + int(args.num_det_px/2)

    shift_y,shift_x = -14, 10

    real_im = np.load(f'{args.reference_file}')[args.ref_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
    real_im = real_im.astype(np.float32)

    # Inject synthetic planet to measure SNR and signal loss
    # Simulates a planet at the given brightness and position to inject into real data
    # Can be set to zero flux to track only the SNR of the real planet
    if args.track_injected_planet: 
        science_noisemap = np.load(f'{args.measurement_noisemap}')[args.sci_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
        real_im_noisemap = science_noisemap.astype(np.float32)
        posxlist, posylist, fluxeslist = generate_positions_sigma(args.inject_how_many_random, args.inject_how_many_sigma_flux,real_im_noisemap, args.num_det_px,scitargname=args.sci_targ_name,data_dir=args.data_dir, inj_pos=np.array([[args.inj_xpos, args.inj_ypos]]))
        original_injected_peaks = []
        injected_planets = []
        for j in range(args.inject_how_many_random):
            injected_companion = {'pos_x_px': posxlist[j], 'pos_y_px': posylist[j], 'flux': fluxeslist[j]}
            with torch.no_grad():
                offset_planet = nn.Parameter(torch.FloatTensor([injected_companion['pos_x_px'] * arcsec2rad(psf_pixel_scale_arcsec), injected_companion['pos_y_px'] * arcsec2rad(psf_pixel_scale_arcsec)]))        
                # Set up the wavefront objects
                wavefronts_list_planet = BroadbandWavefront(wf_npix, diameter, wlen_weights[0], peak_flux_star, offset_planet).to(DEVICE)
                prop_models_planet = [PointPropagate(aperture, lyot, fpm, nircam_OPD, prop_args,oversample=args.oversample, num_det_px=args.num_det_px).to(DEVICE) for _ in range(args.num_ints)]
                pred_planet = [p_model(wavefronts_list_planet, wfe_batch, wlen_weights[1], wlen_weights[0]) for p_model in prop_models_planet]
                pred_planet = torch.mean(torch.cat(pred_planet, 0), 0)
            
            pred_planet_np = pred_planet.cpu().numpy()
            print('injected companion flux is ', injected_companion['flux'])
            print('np.nanmax(pred_planet_np) flux is ', np.nanmax(pred_planet_np))
            # Planet injected at the right position, now we set its brightness
            # injected_companion['flux'] is calculated in generate_positions_sigma() in terms of N times above the Poisson noise limit, 
            # but that is the number required AFTER low pass filtering with sigma=1 px
            # So here, we compute what flux of the pre-smoothed planet, so that after the same low pass filter it results in the desired flux
            smoothed_injected = nan_gaussian_filter(pred_planet_np, 1) # smooth injection with sigma=1 pixel
            
            print('np.nanmax(smoothed_injected) flux is ', np.nanmax(smoothed_injected))
            normaltosmooth = np.nanmax(pred_planet_np) / np.nanmax(smoothed_injected)
            smoothed_conversiontorequiredflux = np.nanmax(smoothed_injected) * injected_companion['flux'] # Dividing the smoothed planet by this number gives the correct flux
            print('smoothed_conversiontorequiredflux flux is ', smoothed_conversiontorequiredflux)
            curr_pred_planet_scaled = pred_planet_np / np.nanmax(pred_planet_np) * injected_companion['flux'] * normaltosmooth# So we divide the pre-smoothed planet by the same flux conversion factor
            print('np.nanmax(curr_pred_planet_scaled) flux is ', np.nanmax(curr_pred_planet_scaled))
            curr_pred_planet_scaled = curr_pred_planet_scaled[None]
            injected_planets.append(curr_pred_planet_scaled)
            original_injected_peaks.append(np.nanmax(curr_pred_planet_scaled))

            # Save ideal injected planet plot
            plt.figure(figsize=[5,5])

            plt.imshow(np.squeeze(curr_pred_planet_scaled), origin='lower')
            plt.title(f'Sum, peak = {np.nansum(curr_pred_planet_scaled):.2f},{np.nanmax(curr_pred_planet_scaled):.2f} (sim planet)')

            plt.tight_layout()
            plt.savefig(f'{vis_dir}/vis_INJECTEDPLANET_{j}.png')
            plt.close()

        original_injected_peaks = np.array(original_injected_peaks)

        if args.inject_the_same_in_all_frames or args.inject_all_in_same_frame:
            injected_planets = merge_injected_planets(np.array(injected_planets))

    if real_im.ndim == 2:
        real_im = real_im[None]

    if args.track_injected_planet: 
        real_im += 0.
        vis_dir_iterations = vis_dir+'/ITERATIONS'
        os.makedirs(vis_dir_iterations, exist_ok=True)

    plt.imsave(f'{vis_dir}/vis_measurement.png', real_im[0], cmap='viridis', origin='lower')
    reference = real_im
    reference = torch.from_numpy(reference).to(DEVICE)

    # Optional median scaling: we find better results by NOT scaling by the median
    if not args.no_median:
        ref_scaled = reference / reference.median()
    else: 
        ref_scaled = reference


    real_im = np.load(f'{args.measurement_file}')[args.sci_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
    real_im = real_im.astype(np.float32)

    if real_im.ndim == 2:
        real_im = real_im[None]
    
    if args.track_injected_planet: 

        if args.inject_all_in_same_frame:

            injected_planets = np.squeeze(injected_planets)
            real_im_planetless= real_im.copy()
            real_im_planetless = torch.from_numpy(real_im_planetless).to(DEVICE)
            # Inject the planets
            if injected_planets.ndim == 2:
                real_im += injected_planets[None]
            elif injected_planets.ndim == 3:
                real_im += injected_planets

        else:
            real_im = np.repeat(real_im, args.inject_how_many_random+1, axis=0)
            # Inject the planets
            for j in range(args.inject_how_many_random):
                real_im[j+1] += injected_planets[j][0]

        sci_injected_peaks = []
        sci_iters = []
        sci_signal_loss = []
        sci_snr = []
    
    observations = torch.from_numpy(real_im).to(DEVICE)
    
    if not args.no_median:
        obs_scaled = observations / observations.median()
    else:
        obs_scaled = observations

    # Load fundamental noise maps 
    if args.measurement_noisemap is not None and args.weigh_loss_by_noise:
        real_im_noisemap = np.load(f'{args.measurement_noisemap}')[args.sci_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
        real_im_noisemap = real_im_noisemap.astype(np.float32)
        if real_im_noisemap.ndim == 2:
            real_im_noisemap = real_im_noisemap[None]
        
        real_im_noisemap = torch.from_numpy(real_im_noisemap).to(DEVICE)
        
        if not args.no_median:
            obs_scaled_noisemap = real_im_noisemap / real_im_noisemap.median()
        else:
            obs_scaled_noisemap = real_im_noisemap

    if args.reference_noisemap is not None:
        reference_im_noisemap = np.load(f'{args.data_dir}/real_data/{args.reference_noisemap}')[args.sci_which_int, (edge1-shift_y):(edge2-shift_y), (edge1-shift_x):(edge2-shift_x)]
        reference_im_noisemap = reference_im_noisemap.astype(np.float32)
        if reference_im_noisemap.ndim == 2:
            reference_im_noisemap = reference_im_noisemap[None]
        
        reference_im_noisemap = torch.from_numpy(reference_im_noisemap).to(DEVICE)
        
        if not args.no_median:
            ref_scaled_noisemap = reference_im_noisemap / reference_im_noisemap.median()
        else:
            ref_scaled_noisemap = reference_im_noisemap
    
    # Visualize the subtracted PSF using the initial WFE with error
    with torch.no_grad():
        pred = [prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
        pred = torch.mean(torch.cat(pred, 0), 0)
    pred_np = pred.cpu().numpy()
    plt.imsave(f'{vis_dir}/vis_PSF_render_init.png', pred_np, cmap='viridis', origin='lower')

    # Option to use simulated PSFs instead of injected planets on real data, for testing.
    if args.use_simulated_data:
        with torch.no_grad():
            true_offset = torch.clone(offset_STAR)
            true_offset[0]+=2e-7
            true_offset[1]+=1.1e-7

            wavefronts_list_simulated = [Wavefront(wf_npix, diameter, wl, peak_flux_star*100, true_offset).to(DEVICE) for wl in wlen_weights[0]]
            wfe_sim_interp = wfe_batch_list[0] + (wfe_batch_list[1] - wfe_batch_list[0]) * 0.5
            wfe_sim_interp_series = [wfe_sim_interp + (1e-6), wfe_sim_interp + (2*1e-6)]
            sim = [prop_models[j](wavefronts_list_simulated, wfe_sim_interp_series[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
            sim = torch.mean(torch.cat(sim, 0), 0)

            shot_noise = torch.normal(mean = sim, std = 1. * torch.sqrt(sim)) - sim
            read_noise = torch.normal(mean = 0, std = 42. * torch.ones_like(sim))
            noise = shot_noise + read_noise

            out_noisy = sim + noise
            sim = torch.clamp(out_noisy, min=0.0)

            np.save(f'{vis_dir}/simulated_obs_no_planet.npy',sim.detach().cpu().numpy())
            observations = torch.clone(sim[None])
            observations = shift_image_subpixel(observations)
            if not args.no_median:
                obs_scaled = observations / observations.median()
            else:
                obs_scaled = observations

            reference = torch.clone(sim[None]) 
            reference = shift_image_subpixel(reference)
            if not args.no_median:
                ref_scaled = reference / reference.median()
            else:
                ref_scaled = reference

    # Load pixel mask file, to mask uncorrected bad pixels from our loss function
    if args.px_mask_file is not None:
        px_mask = np.load(f'{args.px_mask_file}')
        if args.num_det_px != 80:
            edge1, edge2 = 40 - int(args.num_det_px/2), 40 + int(args.num_det_px/2)
            px_mask = px_mask[edge1:edge2, edge1:edge2]
        px_mask_numpy = px_mask.copy()
        px_mask = torch.from_numpy(px_mask.astype(np.float32)).to(DEVICE)[None]

    # If there are known artifacts, load them too (not necessary)
    if args.pxartifactsmap_file is not None:
        pxartifactmap = np.load(f'{args.pxartifactsmap_file}')
        if args.num_det_px != 80:
            edge1, edge2 = 40 - int(args.num_det_px/2), 40 + int(args.num_det_px/2)
            pxartifactmap = pxartifactmap[edge1:edge2, edge1:edge2]
        pxartifactmap_numpy = pxartifactmap.copy()
        pxartifactmap = torch.from_numpy(pxartifactmap.astype(np.float32)).to(DEVICE)[None]

    if args.pxartifactsmap_ref_file is not None:
        pxartifactmap_ref = np.load(f'{args.pxartifactsmap_ref_file}')
        if args.num_det_px != 80:
            edge1, edge2 = 40 - int(args.num_det_px/2), 40 + int(args.num_det_px/2)
            pxartifactmap_ref = pxartifactmap_ref[edge1:edge2, edge1:edge2]
        pxartifactmap_ref_numpy = pxartifactmap_ref.copy()
        pxartifactmap_ref = torch.from_numpy(pxartifactmap_ref.astype(np.float32)).to(DEVICE)[None]
        

    if not args.no_median:
        pred_scaled = pred / pred.median()
    else:
        pred_scaled = pred 

    
    est_residual = (obs_scaled - pred_scaled.expand_as(obs_scaled)).detach().cpu().mean(0).numpy()
    plt.imsave(f'{vis_dir}/vis_est_res_init.png', est_residual, cmap='viridis', origin='lower')

    est_ref_residual = (obs_scaled - ref_scaled.expand_as(obs_scaled)).detach().cpu().mean(0).numpy()
    plt.imsave(f'{vis_dir}/vis_ref_res_init.png', est_ref_residual, cmap='viridis', origin='lower')

    if not args.no_median:
        obs_scaled = observations / observations.median()
        pred_scaled = pred / pred.median()
    else:
        obs_scaled = observations
        pred_scaled = pred
    est_residual = (obs_scaled - pred_scaled.expand_as(obs_scaled)).detach().cpu().mean(0).numpy()
    plt.imsave(f'{vis_dir}/vis_est_res_init.png', est_residual, cmap='viridis', origin='lower')


    # Set up the optimizer and scheduler
    
    """
    wfe_offsets: learns to offset the wfe_batch
    fpm_shifts: learns to shift the focal plane mask
    angle_offsets: learns to offset the star incident angle
    lyot_shifts: learns to shift the lyot mask
    nircam_offsets: learns to offset the nircam opd
    """

    optics_params_normal_lr = list()
    optics_params_bigger_lr = list()
    optics_params_even_bigger_lr = list()

    # To keep the right orders of magnitude, we define learning rates in our parameters relative to a global learning rate
    bigger_lr = args.lr*args.big_lr_factor 
    even_bigger_lr = args.lr*args.even_bigger_lr_factor

    for p_model in prop_models:

        optics_params_bigger_lr+=list(p_model.angle_offsets.parameters())
        optics_params_normal_lr+=list(p_model.wfe_offsets.parameters())
        if args.fit_flat_field:
            optics_params_even_bigger_lr+=list(p_model.detector_flat_field.parameters())

        optics_params_normal_lr+=list(p_model.nircam_offsets.parameters()) 
    
    optimizer_argument = [
        {'params': optics_params_normal_lr, 'lr': args.lr},
        {'params': optics_params_bigger_lr, 'lr': bigger_lr},
        {'params': optics_params_even_bigger_lr, 'lr': even_bigger_lr}
    ]

    # If using two optimizers, currently only AdamW is used, but kept for testing
    optimizer_sgd = torch.optim.SGD(optimizer_argument, momentum=0.9)
    optimizer_adam = torch.optim.AdamW(optimizer_argument, weight_decay=0.0)

    scheduler_sgd = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sgd, T_max=args.iters, eta_min=args.lr)
    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=args.iters, eta_min=args.lr)

    # Set up progress tracking 

    progress_arr_reference = []
    opd_vis_arr = []
    opd_vis_offset_arr = []

    # If we wish to fit as more than one integration; 
    # currently we find our best results by assuming a single one,
    # but kept for testing 
    if args.num_ints > 1:
        opd_vis_arr1 = []
        opd_vis_offset_arr_1 = []

    nircam_opd_vis_arr = []
    nircam_opd_vis_arr_1 = []

    angles_offset_res = []
    angles_offset_res_1 = []

    residual_max_arr = []

    progress_arr_target = []

    # If we wish to also fit translations and rotations of the Lyot stop in the coronagraph
    if args.fit_Lyot:
        lyot_arr = []

    if args.fit_flat_field:
        flat_field = []

    progress_fluxcorrection = []

    cutoff_iter = args.ref_cutoff_iter -1

    switch_step = 0

    if args.track_injected_planet:
        mean_snrs_ref = []
        mean_snrs_sci = []
        hip65426_snr = []
        progress_arr_target_planetless = []
        snr_worse_tracker = 0

    tbar = tqdm.tqdm(range(args.iters + 1), mininterval=30.0)

    # Optimization starts
    for i in tbar:
        optimizer = optimizer_sgd if i < switch_step else optimizer_adam
        scheduler = scheduler_sgd if i < switch_step else scheduler_adam

        # Re start optimizer

        optimizer.zero_grad()

        # Forward model PSF with current parameters
        if i == args.iters:
            pred_1 = []
            for j in range(len(prop_models)):
                res, wf = prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0])
                pred_1.append(res)
                wfnumpy = wf[0].detach().cpu().numpy()
                np.save(f'{vis_dir}/last_iteration_TOTALWAVEFRONT_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy',wfnumpy)
        else:
            pred_1 = [prop_models[j](wavefronts_list1, wfe_batch_list[j], wlen_weights[1], wlen_weights[0]) for j in range(len(prop_models))]
        pred_1 = torch.mean(torch.cat(pred_1, 0), 0)[None]

        if not args.no_median:
            pred_scaled = pred_1 / pred_1.detach().median()
        else:
            pred_scaled = pred_1

        if i > cutoff_iter:
            if args.pxartifactsmap_file is not None:
                pred_scaled+=pxartifactmap
        else:
            if args.pxartifactsmap_ref_file is not None:
                pred_scaled+=pxartifactmap_ref

        # Compute difference between PSF model and reference and science observations
        est_residual_obs = obs_scaled - pred_scaled.detach().expand_as(obs_scaled)
        est_residual_ref = ref_scaled - pred_scaled.detach()

        if args.track_injected_planet and args.inject_all_in_same_frame:
            est_residual_obs_planetless = real_im_planetless - pred_scaled.detach().expand_as(real_im_planetless)

        # Compute loss function, masking uncorrected pixels to avoid biasing the optimization
        if args.px_mask_file is not None:
            if args.reference_noisemap is not None:
                global_l1_loss = weighted_smooth_l1_loss(pred_scaled, ref_scaled, ref_scaled_noisemap,reduction='none')
            else:
                global_l1_loss = F.smooth_l1_loss(pred_scaled, ref_scaled, reduction='none')
            
            if args.measurement_noisemap is not None and args.weigh_loss_by_noise:
                obs_l1_loss = weighted_l1_loss(pred_scaled, obs_scaled, obs_scaled_noisemap,reduction='none')
            else:
                obs_l1_loss = F.l1_loss(pred_scaled.expand_as(obs_scaled), obs_scaled, reduction='none')

            global_l1_loss = global_l1_loss * px_mask
            global_l1_loss = global_l1_loss.sum() / px_mask.sum()  

            obs_l1_loss = obs_l1_loss * px_mask.expand_as(obs_scaled) 
            if args.inject_all_in_same_frame:
                obs_l1_loss = obs_l1_loss.sum() / px_mask.sum()
            else:
                obs_l1_loss = obs_l1_loss.sum() / (px_mask.sum() *(args.inject_how_many_random+1))
                obs_l1_loss = obs_l1_loss / (args.inject_how_many_random+1)
        else:
            if args.reference_noisemap is not None:
                global_l1_loss = weighted_smooth_l1_loss(pred_scaled, ref_scaled, ref_scaled_noisemap)
            else:
                global_l1_loss = F.smooth_l1_loss(pred_scaled, ref_scaled)
            
            if args.measurement_noisemap is not None and args.weigh_loss_by_noise:
                obs_l1_loss = weighted_l1_loss(pred_scaled, obs_scaled, obs_scaled_noisemap)
            else:
                obs_l1_loss = F.l1_loss(pred_scaled, obs_scaled)

        # Update model parameters based on the reference star if we are in stage 1 optimization,
        # otherwise use loss function on the science target directly as we are in stage 2
        if i > cutoff_iter:
            if i < (args.stage_fluxpos_cutoff_iter + cutoff_iter):
                loss = obs_l1_loss
            else:
                loss = args.sci_lr_weight * obs_l1_loss
        else:
                loss = global_l1_loss

        # If we are using smoothing regularization (e.g., OPD maps must not have high spatial frequency content)
        if args.smooth is not None:
            opd1 = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])
            if args.num_ints > 1:
                opd2 = prop_models[1].wfe_offsets.forward(wfe_batch_list[1])
            
            TVL = total_variation_loss(opd1)
            if args.num_ints > 1:
                TVL+= total_variation_loss(opd2)
                

            nircam_opd = prop_models[0].nircam_offsets.get_res()
            if args.num_ints > 1:
                nircam_op1 = prop_models[1].nircam_offsets.get_res()
            
            TVL += total_variation_loss(nircam_opd)
            if args.num_ints > 1:
                TVL += total_variation_loss(nircam_op1)

            if i > cutoff_iter:
                loss = loss + args.smooth * TVL
            else:
                loss = loss + args.smooth * TVL

        # If we have more than one integration, optionally 
        # encourage that OPD solutions must be similar to one another 
        if args.num_ints > 1 and args.OPD_loss_weight is not None:
            opd1 = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])
            opd2 = prop_models[1].wfe_offsets.forward(wfe_batch_list[1])


            opd1_scaled = opd1 
            opd2_scaled = opd2 


            opds_disimilarity = F.smooth_l1_loss(opd1_scaled, opd2_scaled)

            if i > cutoff_iter:
                loss = loss + args.OPD_loss_weight * opds_disimilarity * 1e15
            else:
                loss = loss + args.OPD_loss_weight * opds_disimilarity * 1e15

        elif args.OPD_loss_weight is not None:
            opd = prop_models[0].wfe_offsets.forward(wfe_batch_list[0])

            opds_disimilarity_before = F.smooth_l1_loss(opd, wfe_batch_list[0])
            opds_disimilarity_after = F.smooth_l1_loss(opd, wfe_batch_list[1])

            opds_total_disimilarity = opds_disimilarity_before + opds_disimilarity_after

            if i > cutoff_iter:
                loss = loss + args.OPD_loss_weight * opds_total_disimilarity * 1e15
            else:
                loss = loss + args.OPD_loss_weight * opds_total_disimilarity * 1e15

        # Backpropagate loss function
        loss.backward()
        
        # If we are on stage 2:
        if i > cutoff_iter:
            # Keep fixed the OPD before the focal plane mask during the stage 2:
            if args.freeze_OTE_OPD_sci:
                for p_model in prop_models:
                    for t in p_model.wfe_offsets.parameters():
                        if t.requires_grad:
                            t.grad *= 0.0
            # If we are in the first sub-stage, only fit position in the model and update flux, but do not update any other parameters
            if i < (args.stage_fluxpos_cutoff_iter + cutoff_iter):
                # Update flux parameter to match total flux in the observation within a central N pixels
                with torch.no_grad():
                    fluxwindsize = args.fluxwindsize 
                    maskslice = px_mask[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]
                    tot_flux_pred = (maskslice*pred_scaled[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                    tot_flux_target = (maskslice*obs_scaled[:,(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                    if args.inject_all_in_same_frame:
                        flux_mismatch_ratio = tot_flux_target / tot_flux_pred
                    else:
                        flux_mismatch_ratio = (tot_flux_target / (args.inject_how_many_random+1))/ tot_flux_pred
                    for p_model in prop_models:
                        p_model.flux_correction.data *=flux_mismatch_ratio.to(DEVICE)

                    # Keep other parameters fixed in this sub-stage
                    for p_model in prop_models:
                        for t in p_model.wfe_offsets.parameters():
                            if t.requires_grad:
                                t.grad *= 0.0
                        for t in p_model.nircam_offsets.parameters():
                            if t.requires_grad:
                                t.grad *= 0.0
                        if args.enhance_sci_pos_grad is not None: # If we want to bump up the learning rate in the position (useful experimenting with far-from-coronagraph initial conditions)
                            for t in p_model.angle_offsets.parameters():
                                if t.requires_grad:
                                    t.grad *= args.enhance_sci_pos_grad
            else:
                # If we are not in the first sub-stage, do not update position anymore
                if args.freeze_position:
                    for p_model in prop_models:
                        for t in p_model.angle_offsets.parameters():
                            if t.requires_grad:
                                t.grad *= 0.0

        # If we are still in stage 1:
        elif i < args.stage_fluxpos_cutoff_iter:
            # If we are in the first sub-stage, update flux to equate total flux in the observation within a central N pixels
            with torch.no_grad():
                fluxwindsize = args.fluxwindsize 
                maskslice = px_mask[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]
                tot_flux_pred = (maskslice*pred_scaled[:,(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(pred_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(pred_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                if i > cutoff_iter:
                    tot_flux_target = (maskslice*obs_scaled[:,(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(obs_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(obs_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()
                else:
                    tot_flux_target = (maskslice*ref_scaled[:,(int(ref_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(ref_scaled.shape[-1]/2) + int(fluxwindsize/2)),(int(ref_scaled.shape[-1]/2) - int(fluxwindsize/2)):(int(ref_scaled.shape[-1]/2) + int(fluxwindsize/2))]).sum()

                flux_mismatch_ratio = tot_flux_target / tot_flux_pred
                for p_model in prop_models:
                    p_model.flux_correction.data *=flux_mismatch_ratio.to(DEVICE)
            # Do not update other parameters while we are updating position and flux only
            for p_model in prop_models:
                for t in p_model.wfe_offsets.parameters():
                    if t.requires_grad:
                        t.grad *= 0.0
                for t in p_model.nircam_offsets.parameters():
                    if t.requires_grad:
                        t.grad *= 0.0
        else:
            # If we are no longer in the first sub-stage, do not update position anymore
            if args.freeze_position:
                for p_model in prop_models:
                    for t in p_model.angle_offsets.parameters():
                        if t.requires_grad:
                            t.grad *= 0.0
        
        # In the specified iteration range, optimize the detector response per pixel
        if i < args.stage_fit_flat_fields_start_iter or i > args.stage_fit_flat_fields_end_iter:
            for t in p_model.detector_flat_field.parameters():
                if t.requires_grad:
                    t.grad *= 0.0

        else:
            for t in p_model.detector_flat_field.parameters():
                if t.requires_grad:
                    t.grad *= 1.0

        # Periodically save flat field solution
        if i % 100 ==0 and args.fit_flat_field:
            flat_field.append(prop_models[0].detector_flat_field.get_res().squeeze().detach().cpu().numpy())

        optimizer.step()
        scheduler.step()

        # Now, track the signal loss and SNR of injected planets (and real planets, if applicable)
        if i%300 == 0 and i > cutoff_iter:
            if args.track_injected_planet: 
                # Exclude bad pixels from SNR calculation
                if args.px_mask_file is not None:
                    px_mask_nanned = np.where(px_mask_numpy == 0., np.nan, px_mask_numpy)
                else:
                    px_mask_nanned = 1.
                if i%300 == 0:
                    vis_dir_iterations_current = os.path.join(vis_dir_iterations, 'iter'+str(i))
                    os.makedirs(vis_dir_iterations_current, exist_ok=True)
                if i > cutoff_iter:
                    label='science'
                    residual = est_residual_obs.detach().cpu().numpy()
                    if args.sci_targ_name == 'HIP65426':
                        if args.sci_targ_name == 'HIP65426': # Known position of HIP 65426 b
                            x_pos, y_pos = -7.2 * psf_pixel_scale_arcsec, 11 * psf_pixel_scale_arcsec # hand tuned for HIP 65426 in the pre-rotated frame
                            sourcemaskrad =1.3
                        residual = np.squeeze(residual)
                        if residual.ndim == 2:
                            myres = residual
                            residual = residual[None]
                        elif residual.ndim==3:
                            myres = residual[0]

                        masked_slices = []
                        for j in range(len(residual)):
                            masked_slices.append(circular_mask(residual[j], x_pos, y_pos, sourcemaskrad, np.nan))
                        masked_residual = np.array(masked_slices)
                    else:
                        masked_residual = residual
                else:
                    label='reference'
                    residual = est_residual_ref 
                    residual = residual.detach().cpu().numpy()
                    residual = np.squeeze(residual)
                    masked_residual = residual

                if args.px_mask_file is not None:
                    px_mask_nanned = np.where(px_mask_numpy == 0., np.nan, px_mask_numpy)
                else:
                    px_mask_nanned = 1.
                
                curr_iter_snrs, curr_iter_signal_blob_peak = [], []
                #Check SNRs of known synthetic planets that were injected
                for j in range(args.inject_how_many_random):
                    injected_companion = {'pos_x_px': posxlist[j], 'pos_y_px': posylist[j], 'flux': fluxeslist[j]}
                    if args.inject_all_in_same_frame:
                        curr_frame_orig = residual[0]
                        curr_frame_masked = masked_residual[0]
                    else:
                        curr_frame_orig = residual[j+1]
                        curr_frame_masked=masked_residual[j+1]
                    _, annulus, signal_blob, peak_blob = calc_snr(curr_frame_masked*px_mask_nanned, injected_companion['pos_x_px']*psf_pixel_scale_arcsec, injected_companion['pos_y_px']*psf_pixel_scale_arcsec,0.3,np.sqrt(injected_companion['pos_x_px']**2 + injected_companion['pos_y_px']**2)*psf_pixel_scale_arcsec, blur_annulus=args.blur_annulus_SNR, blur_before_signal=args.blur_before_signal,return_signal_peak=True)
                    curr_snr, _, _ = calc_snr(curr_frame_masked*px_mask_nanned, injected_companion['pos_x_px']*psf_pixel_scale_arcsec, injected_companion['pos_y_px']*psf_pixel_scale_arcsec,0.3,np.sqrt(injected_companion['pos_x_px']**2 + injected_companion['pos_y_px']**2)*psf_pixel_scale_arcsec, blur_annulus=args.blur_annulus_SNR, blur_before_signal=args.blur_before_signal, sigma_kernel=1)
                    curr_iter_snrs.append(curr_snr)
                    curr_iter_signal_blob_peak.append(peak_blob)
                    if args.sci_targ_name == 'HIP65426':
                        _, hip_annulus, hip_signal_blob = calc_snr(myres*px_mask_nanned, x_pos, y_pos,sourcemaskrad,np.sqrt(x_pos**2 + y_pos**2),width=0.5, blur_annulus=args.blur_annulus_SNR, blur_before_signal=args.blur_before_signal)
                        hip_curr_snr, _, _ = calc_snr(myres*px_mask_nanned, x_pos, y_pos,sourcemaskrad,np.sqrt(x_pos**2 + y_pos**2),width=0.5, blur_annulus=args.blur_annulus_SNR, blur_before_signal=args.blur_before_signal, sigma_kernel=1)
                        hip65426_snr.append(hip_curr_snr)

                    if i%300 == 0:
                        plt.figure(figsize=[15,10])
                        plt.subplot(231)
                        plt.imshow(curr_frame_orig, origin='lower')
                        plt.title(f'Sum, peak = {np.nansum(curr_frame_orig):.2f},{np.nanmax(curr_frame_orig):.2f} (orig)')
                        plt.subplot(232)
                        plt.imshow(curr_frame_masked, origin='lower', vmin=-10)
                        plt.subplot(234)
                        plt.imshow(annulus, origin='lower')
                        plt.title(f'Stddev = {np.nanstd(annulus):.2f} (annulus synthetic)')
                        plt.subplot(235)
                        plt.imshow(signal_blob, origin='lower')
                        plt.title(f'Sum, peak = {np.nansum(signal_blob):.2f},{peak_blob:.2f},\n pos x, pos y, flux input ={injected_companion["pos_x_px"]:.2f},{injected_companion["pos_y_px"]:.2f},{injected_companion["flux"]:.2f}')
                        whole_annulus = np.where(~np.isnan(signal_blob), signal_blob, annulus)
                        plt.subplot(236)
                        plt.imshow(whole_annulus, origin='lower')
                        plt.title(f'Whole annulus. \n Peak, mean, stddev = {np.nanmax(whole_annulus):.2f}, {np.nanmean(whole_annulus):.2f}, {np.nanstd(whole_annulus):.2f}')
                        plt.suptitle(f'Curr SNR: {curr_snr:.2f}')
                        plt.tight_layout()
                        plt.savefig(f'{vis_dir_iterations_current}/vis_measurement_INJECTED_iter{i}_{label}_comp{j}.png')
                        plt.close()

                curr_iter_snrs = np.array(curr_iter_snrs)
                curr_iter_signal_blob_peak = np.array(curr_iter_signal_blob_peak)

                if i > cutoff_iter:
                    sci_injected_peaks.append(curr_iter_signal_blob_peak)
                    sci_signal_loss.append(curr_iter_signal_blob_peak/original_injected_peaks)
                    mean_snrs_sci.append(np.nanmean(curr_iter_snrs)) # this will be the main thing tracked; the rest is mainly for debugging or visualization
                    sci_snr.append(curr_iter_snrs)
                    sci_iters.append(i)

        # Save some progress
        if i == args.forced_stop_iter or i%500 == 0:
            if i > cutoff_iter:
                progress_arr_target.append(est_residual_obs[0].cpu().numpy())
                if args.inject_all_in_same_frame:
                    progress_arr_target_planetless.append(est_residual_obs_planetless[0].cpu().numpy())
            else:
                progress_arr_target.append(est_residual_obs[0].cpu().numpy())
                progress_arr_reference.append(est_residual_ref[0].cpu().numpy())

            
            if i % args.vis_freq == 0 and i > cutoff_iter:
                plt.imsave(f'{vis_dir}/vis_est_res_TARG_{i}.png', progress_arr_target[-1], cmap='viridis', origin='lower')
            elif i % args.vis_freq == 0:
                plt.imsave(f'{vis_dir}/vis_est_res_REF_{i}.png', progress_arr_reference[-1], cmap='viridis', origin='lower')

            
            cur_opd = prop_models[0].wfe_offsets.forward(wfe_batch_list[0]).squeeze().detach().cpu()
            opd_vis_arr.append(cur_opd)

        if args.fit_Lyot:
            with torch.no_grad():
                result= prop_models[0].lyot_shifts(prop_models[0].lyot)
                lyot_arr.append(result.detach().cpu())
        if i == args.forced_stop_iter or i%500 == 0:
            opd_vis_offset_arr.append(prop_models[0].wfe_offsets.get_res().squeeze().detach().cpu())
            
            angles_offset_res.append(prop_models[0].angle_offsets().squeeze().detach().cpu())
            progress_fluxcorrection.append(prop_models[0].flux_correction(1.))
            if args.num_ints > 1:
                cur_opd = prop_models[1].wfe_offsets.forward(wfe_batch_list[1]).squeeze().detach().cpu()
                opd_vis_arr1.append(cur_opd)

                angles_offset_res_1.append(prop_models[1].angle_offsets())

                opd_vis_offset_arr_1.append(prop_models[1].wfe_offsets.get_res().squeeze().detach().cpu())
                curr_nircam_opd_1 = prop_models[1].nircam_offsets.get_res().squeeze().detach().cpu()
                nircam_opd_vis_arr_1.append(curr_nircam_opd_1)

            curr_nircam_opd = prop_models[0].nircam_offsets.get_res().squeeze().detach().cpu()
            nircam_opd_vis_arr.append(curr_nircam_opd)

        
        tbar_out = {'loss': global_l1_loss.item()}
        tbar.set_postfix(tbar_out)

        # Stop optimization loop if a stopping point earlier than the scheduler is wanted
        if args.forced_stop_iter is not None:
            if i == args.forced_stop_iter:
                break

    # End of loop; now save results 
    if args.track_injected_planet: 
        # Save results for the planet injection every certian number of iterations; useful to define a stopping condition afterwards
        sci_injected_peaks = np.array(sci_injected_peaks)
        sci_signal_loss = np.array(sci_signal_loss)
        hip65426_snr = np.array(hip65426_snr)
        sci_snr = np.array(sci_snr)
        sci_iters = np.array(sci_iters)

        np.save(os.path.join(vis_dir_iterations, 'sci_injected_peaks_postsub.npy'), sci_injected_peaks)
        np.save(os.path.join(vis_dir_iterations, 'sci_injected_signal_loss.npy'), sci_signal_loss)
        np.save(os.path.join(vis_dir_iterations, 'realtarget_snr.npy'), hip65426_snr)
        np.save(os.path.join(vis_dir_iterations, 'sci_injected_snr.npy'), sci_snr)
        np.save(os.path.join(vis_dir_iterations, 'sci_iters.npy'), sci_iters)


    visvidfreq = 1
    # Video animation of stage 1 optimization
    progress_arr = np.array(progress_arr_reference)[::visvidfreq]
    progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    progress_arr = np.flip(progress_arr, 1)
    imageio.mimsave(f'{vis_dir}/progress_REFERENCE.mp4', progress_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })
    

     # Video animation of stage 2 optimization
    progress_arr = np.array(progress_arr_target)[::visvidfreq]
    progress_arr = np.array([(im - im.min()) / (im.max() - im.min()) for im in progress_arr])
    progress_arr = np.uint8(cm.viridis(progress_arr) * 255)
    progress_arr = np.flip(progress_arr, 1)
    imageio.mimsave(f'{vis_dir}/progress_TARGET.mp4', progress_arr, 
                    'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 30, })

    # Save result on reference and science targets
    target_numpys = np.squeeze(np.array(progress_arr_target))
    reference_numpys = np.squeeze(np.array(progress_arr_reference))

    np.save(f'{vis_dir}/last_iteration_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy', target_numpys[-1])

    np.save(f'{vis_dir}/last_iteration_REFERENCE_oversample_{args.oversample}_wl_sampling_{args.num_wl}.npy', reference_numpys[-1])
    
   

