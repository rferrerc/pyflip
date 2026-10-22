# testing.py with diffusion priors on the OTE / Lyot OPDs and a star flux fitted by gradient.
# The default values represent the best run so far (SNR ~52 for HPI 65426)
# To get the default behaviour, set USE_PRIOR = False, lyot_mode = 'free', fit_flux = False
# Configuration:
# * USE_PRIOR injects the prior behaviour onto the pyFLIP code
# * ote_prior_mode ('offset' | 'replace') for whether the prior represents a full WFS map or the drift between consecutive WFS maps
# Inference/diffusion schedule config
# * ref_ote_prior_start/end (DPS window on the reference frame), sci_ote_prior_start/end (window on the science frame),
# * ote_grid_frozen_until, ote_guidance_scale,
# * ote_prior_sigma_max/min, ote_prior_sde, ote_prior_seed,
# Similar parameters for the lyot:
# * lyot_mode ('free' | 'drift'), lyot_prior_start/end,
# * lyot_sci_prior_start/end, lyot_guidance_scale, lyot_sci_guidance_scale
# Extra module to fit the flux scaling:
# * fit_flux, flux_lr.
#
# The diff between this file and tesing.py is small

from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from optimization_handlers import default_stages
from priors.model import OPDPrior
from priors.dpsadam import DPSAdamOPD, DPSPhase
from priors.stages import prior_stages
from priors.flux import FluxScale
import torch.nn as nn
import torch 
from dl_utils import load_model_data, load_and_crop_data, inject_fake_planets, hardcoded_match_flux,track_SNR
import numpy as np
import tqdm
import torch.nn.functional as F

### TODO:
#- Track SNR
#- save calibrated model for future reference
#- Visualization tools
# Test on other data

#### PRIOR SETTINGS

priors_dir = '../HIP65426/priors'
# The two priors (checkpoint, training statistics, network hyperparameters of the checkpoint): on full OTE OPD maps (used in
# mode 'replace') and on the drift between consecutive WSS maps (mode 'offset', the Lyot plane, and any window on the science frame)
ote_map_prior = dict(ckpt_path=priors_dir + '/ote_prior_ckpt_47500.pt', stats_path=priors_dir + '/ote_prior_stats.npz', kind='map',
                     net_kwargs=dict(img_resolution=256, img_channels=1, model_channels=64, channel_mult=[1, 1, 2, 2, 4], attn_resolutions=[16], num_blocks=1))
ote_drift_prior = dict(ckpt_path=priors_dir + '/ote_drift_prior_ckpt_40000.pt', stats_path=priors_dir + '/ote_prior_stats.npz', kind='drift',
                       normalization_path=priors_dir + '/ote_drift_prior_normalization.npz',
                       net_kwargs=dict(img_resolution=256, img_channels=1, model_channels=64, channel_mult=[1, 1, 2, 2, 4], attn_resolutions=[16], num_blocks=2))

# OTE OPD (priors/dpsadam.py): the WFS map as in testing.py before ref_ote_prior_start, DPS-Adam between ref_ote_prior_start
# and ref_ote_prior_end (one diffusion step per iteration), frozen afterwards; a residual free grid (testing.py's) is fitted
# from ote_grid_frozen_until on (None: from ref_ote_prior_end; past the last iteration = never).
USE_PRIOR = True
ote_prior_mode = 'offset' # 'offset': the drift prior's output is added to the WFS map; 'replace': the map prior models the full OTE OPD and the WFS map is ignored
ref_ote_prior_start, ref_ote_prior_end = 600, 5500
ote_grid_frozen_until = 11001
# Optional second DPS window on the science frame with the drift prior (a drift on top of what the reference stage gave);
# the residual grid is frozen during it and refined afterwards (stage 2.2, priors/stages.py). None = off.
sci_ote_prior_start, sci_ote_prior_end = None, None # e.g. 6100, 8100
ote_guidance_scale = 5e-3 # learning rate of the latent in the optimizer (network units; 1 unit = 2 std of the training maps = 47.5 nm)
ote_prior_sigma_max, ote_prior_sigma_min = 10., 0.002 # noise schedule
ote_prior_sde = True # reverse SDE (noise at each step) or probability-flow ODE
ote_prior_seed = 0

# stpsf's fixed field-dependent OTE WFE at the coronagraph field point (30 nm rms; InverseBench/scripts/make_ote_field_term.py).
# The WFS map of the data folder already contains it (stpsf adds it when the OTE plane is exported at the coronagraph
# aperture); the prior's training maps are at the sensing field point and do not. Needed in 'replace' mode only.
ote_field_term_path = priors_dir + '/ote_field_term_NRCA5_MASK335R.npy'

# Lyot-plane (NIRCam) OPD: 'free' (testing.py: free pixel grid) or 'drift' (the drift prior in mode 'offset' on top of the static
# NIRCam OPD; DPS window lyot_prior_start-lyot_prior_end on the reference frame and, if set, lyot_sci_prior_start-lyot_sci_prior_end
# on the science frame; same schedule as the OTE; the free grid is never fitted). Guidance scales None = ote_guidance_scale.
lyot_mode = 'drift'
lyot_prior_start, lyot_prior_end = 600, 5500
lyot_sci_prior_start, lyot_sci_prior_end = 6100, 11000
lyot_guidance_scale, lyot_sci_guidance_scale = None, 1.5e-2

# Star flux: testing.py only rescales it to the data in stages 1.1 and 2.1 (hardcoded_match_flux, counts in the central 30 px);
# fit_flux also fits a relative flux scale by gradient (priors/flux.py; own optimizer group, lr flux_lr = relative step) in
# every other stage, so that the model can dim the halo without scattering light out of the field with high-frequency OPD
fit_flux = True
flux_lr = 1e-3

#### MODEL GENERATION


peak_flux_star  = nn.Parameter(torch.FloatTensor([1.]))

wf_npix = 1024
diameter = 6.603464
DEVICE = 'cuda'
num_wl = 21
oversample = 2
data_dir = priors_dir + '/model_setup'
out_dir = priors_dir + '/outputs'
num_det_px = 80

# Stellar position
offset_STAR = nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE)

# TODO: wrapper function to do all of this at once for the simpler cases, including using STPSF to generate the masks and such
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir)

# Set up the wavefront object: this will be propagated by our optical model
wavefront_broad = BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR).to(DEVICE)
if fit_flux:
    wavefront_broad.flux_correction = FluxScale(DEVICE=DEVICE)

diffnircam = assemble_JWST_NIRCam_coron(data_dir,num_det_px)

# The priors replace the free OPD grids (OPDOffsetModule) of the primary / Lyot stop. They keep a 'grid' parameter (the
# residual free grid), so the optimizer groups and stages below work unchanged.
ote_prior, lyot_prior = None, None
if USE_PRIOR:
    field_term = None
    if ote_prior_mode == 'replace':
        field_term = torch.from_numpy(np.flip(np.load(ote_field_term_path), axis=0).copy())[None] # flipped like the WFS map in load_model_data
    ref_prior = OPDPrior(**(ote_map_prior if ote_prior_mode == 'replace' else ote_drift_prior), DEVICE=DEVICE)
    phases = [DPSPhase(ref_prior, ref_ote_prior_start, ref_ote_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde, seed=ote_prior_seed, DEVICE=DEVICE)]
    if sci_ote_prior_start is not None:
        sci_prior = ref_prior if ref_prior.kind == 'drift' else OPDPrior(**ote_drift_prior, DEVICE=DEVICE)
        phases.append(DPSPhase(sci_prior, sci_ote_prior_start, sci_ote_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde,
                               seed=None if ote_prior_seed is None else ote_prior_seed + 1, DEVICE=DEVICE))
    ote_prior = DPSAdamOPD(phases, mode=ote_prior_mode, field_term=field_term, grid_frozen_until=ote_grid_frozen_until, DEVICE=DEVICE)
    diffnircam.optical_system.JWSTPrimary.wfe_offsets = ote_prior
if lyot_mode == 'drift':
    drift_prior = ote_prior.phases[0].prior if USE_PRIOR and ote_prior.phases[0].prior.kind == 'drift' else OPDPrior(**ote_drift_prior, DEVICE=DEVICE)
    lyot_phases = [DPSPhase(drift_prior, lyot_prior_start, lyot_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde,
                            seed=None if ote_prior_seed is None else ote_prior_seed + 2, DEVICE=DEVICE)]
    if lyot_sci_prior_start is not None:
        lyot_phases.append(DPSPhase(drift_prior, lyot_sci_prior_start, lyot_sci_prior_end, sigma_max=ote_prior_sigma_max, sigma_min=ote_prior_sigma_min, sde=ote_prior_sde,
                                    seed=None if ote_prior_seed is None else ote_prior_seed + 3, DEVICE=DEVICE))
    lyot_prior = DPSAdamOPD(lyot_phases, mode='offset', grid_frozen_until=float('inf'), DEVICE=DEVICE) # the free grid is never fitted
    diffnircam.optical_system.NIRCamLyotstop.wfe_offsets = lyot_prior

result = diffnircam(wavefront_broad)

result = result.detach().cpu().numpy()

np.save(out_dir + '/res2.npy', result)


#### OPTIMIZER SETUP

scheduler_iters = 20000 # Max number of iterations for LR scheduler: we can halt the optimization before but not after this

# Choose parameters that are going to be optimized, and assign them reasonable learning rates

# OPD in priamry and post Lyot stop
opds = [
    diffnircam.optical_system.JWSTPrimary.wfe_offsets.grid,
    diffnircam.optical_system.NIRCamLyotstop.wfe_offsets.grid
]

# Star position offset from coronagraph center
star_position = [
    wavefront_broad.angles_offset
]

# Detector flats 
detector_flats = [
    diffnircam.optical_system.NIRCamdetector.detector_flat_field.grid
]

# The 4 works well for many datasets but could be tweaked in some cases; e.g. we have found it needs to be
# smaller for fainter stars sometimes
optimizer_argument = [
        {'params': opds, 'lr': 4e-9},
        {'params': star_position, 'lr': 4e-8},
        {'params': detector_flats, 'lr': 4e-6}
    ]
# The likelihood step of DPS-Adam is this optimizer's step on the latents (learning rate = guidance scale, one group per window)
if USE_PRIOR:
    optimizer_argument.append(ote_prior.param_group(ote_guidance_scale))
if lyot_prior is not None:
    lyot_scales = [ote_guidance_scale if lyot_guidance_scale is None else lyot_guidance_scale, ote_guidance_scale if lyot_sci_guidance_scale is None else lyot_sci_guidance_scale]
    optimizer_argument += [{'params': [lat], 'lr': lr} for lat, lr in zip(lyot_prior.latents, lyot_scales)]
if fit_flux:
    optimizer_argument.append({'params': [wavefront_broad.flux_correction.log_scale], 'lr': flux_lr})

optimizer = torch.optim.AdamW(optimizer_argument, weight_decay=0.0)

# Make eta_min the same as OPDs LR, we just find it does well, but we might want something different at some point
# because making all parameters have the same LR as the minimum limit may be sub-optimal

# TODO: we might want to get rid fo the scheduler altogether at some point, or 
# test how the eta_min affects parameters with different learning rates
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_iters, eta_min=4e-9) 


######### LOAD RELEVANT DATA
refdatapath = '../HIP65426/drive_data/HIP65426/jw01386001001_0310e_00001_nrcalong_calints.fits'
scidatapath = '../HIP65426/drive_data/HIP65426/jw01386002001_0310a_00001_nrcalong_calints.fits'
ref, ref_noise = load_and_crop_data(refdatapath, num_det_px=80, coron_type='MASK335R')
sci, sci_noise = load_and_crop_data(scidatapath, num_det_px=80, coron_type='MASK335R')

# Array with 0s and 1s, used to mask certain pixels (e.g., dead or hot pixels) from contributing to the loss function
# Has to be provided by the user, probably after a first pass. Will depend on the pixel cleaning routine used.
# It can also be used to mask out full regions of the image from the optimization, for arbitrary testing.
px_mask = np.load('../HIP65426/model_setup/pixel_masks/pixel_masks.npy')
px_mask = px_mask.astype(np.float32)
px_mask_numpy = px_mask.copy()
px_mask = torch.from_numpy(px_mask).to(DEVICE)[None]

### Inject planet(s) and such to track SNR
# Only needed on science target really

flux_n_5_sigma = 5 # inject planet 5 times brighter than the 5 sigma photon noise limit

fakes = [{'pos_x_px':10., 'pos_y_px': 0.}] # fake planet locations, relative to center of the array, in px

sci, injected_fluxes = inject_fake_planets(sci,sci_noise, wavefront_broad, diffnircam,fakes, flux_n_5_sigma, smooth_sigma_for_flux=1.)

ref = torch.from_numpy(ref).to(DEVICE)
sci = torch.from_numpy(sci).to(DEVICE)

####### RUN OPTIMIZATION
sci_loss_weight = 0.005 # How much to weigh down the loss function when evaluating on science target relative to reference target; essentially make optimization slower

measure_SNR_every = 300 # measure the SNR of the fake planet(s) every how many iterations 

stage_cutoffs = [600, 3500, 5500, 6100, 11000] # Iterations separating stage 1.1, 1.2, 1.3, 2.1 and 2.2 in our default stage framework


tbar = tqdm.tqdm(range(scheduler_iters + 1), mininterval=30.0)

# Track injected planets
signal_peaks_arr, signal_losses_arr, snrs_arr, track_iters = [], [], [], []

for i in tbar:

    # Re start optimizer at the start always (IMPORTANT!)
    optimizer.zero_grad()

    # Prior move of the latents (previous diffusion step) and denoised OPD layers for this iteration
    if USE_PRIOR:
        ote_prior.step(i)
    if lyot_prior is not None:
        lyot_prior.step(i)

    # Model new PSF
    PSF = diffnircam(wavefront_broad)

    if i < stage_cutoffs[2]: # Iteration dividing stage 1.3 from 2.1: go from calibration on reference to fitting science
        loss = F.smooth_l1_loss(PSF, ref, reduction='none') 
    else:
        loss = F.l1_loss(PSF, sci, reduction='none') * sci_loss_weight

    loss *= px_mask # Mask out bad pixels so they don't contribute to loss function
    loss = loss.sum() / px_mask.sum() # Collapse loss into a number

    # At this step, we can add some regularization to the loss function if we want. E.g., smoothness or whatever
    # Otherwise, continue to backpropagation and parameter update

    # Backpropagate loss function, populate parameter gradients
    loss.backward()

    if i < stage_cutoffs[0]: # TODO: fix flux fitting with gradient; then this will not be necessary
        wavefront_broad = hardcoded_match_flux(PSF,ref,px_mask,wavefront_broad)
    elif i > stage_cutoffs[2] and i < stage_cutoffs[3]:
        wavefront_broad = hardcoded_match_flux(PSF,sci,px_mask,wavefront_broad)
    if fit_flux and (i <= stage_cutoffs[0] or stage_cutoffs[2] <= i <= stage_cutoffs[3]):
        wavefront_broad.flux_correction.log_scale.grad *= 0. # flux by gradient only in the stages without the hardcoded rescaling

    if USE_PRIOR:
        prior_stages(diffnircam, wavefront_broad, i, stage_cutoffs=stage_cutoffs, fit_ote_in_science=sci_ote_prior_start is not None) # default_stages, plus the OTE grid refined in stage 2.2 after a science DPS window
    else:
        default_stages(diffnircam, wavefront_broad, i, stage_cutoffs=stage_cutoffs) # Zero-out (in-place) gradients of parameters we are not fitting at a given stage

    # Update parameters 
    optimizer.step()
    scheduler.step()

    if i == stage_cutoffs[2] -1:
        # End of calibration!
        residual_ref = ref - PSF 
        residual_ref = residual_ref.detach().cpu().numpy()
        np.save(out_dir + '/residual_ref.npy', residual_ref)

    if i == stage_cutoffs[-1]:
        # End! 
        residual_sci = sci - PSF 
        residual_sci = residual_sci.detach().cpu().numpy()
        np.save(out_dir + '/residual_sci.npy', residual_sci)
        break
        


    # Now, track SNR of injected planet(s)
    if i%measure_SNR_every == 0 and i > stage_cutoffs[2]: # only measure in science target
        residual_sci = sci - PSF 
        residual_sci = residual_sci.detach().cpu().numpy()
        signal_peaks, signal_losses, snrs = track_SNR(residual_sci, i, fakes, injected_fluxes,mask_name='MASK335R', snr_smooth_sigma=1., px_mask_numpy=px_mask_numpy)
        signal_peaks_arr.append(signal_peaks)
        signal_losses_arr.append(signal_losses)
        snrs_arr.append(snrs)
        track_iters.append(i)

# After everything, save these diagnostics and such 
signal_peaks_arr = np.array(signal_peaks_arr)
signal_losses_arr = np.array(signal_losses_arr)
snrs_arr = np.array(snrs_arr)
track_iters = np.array(track_iters)
