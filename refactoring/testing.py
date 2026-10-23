from wavefronts import BroadbandWavefront
from model_building import assemble_JWST_NIRCam_coron
from optimization_handlers import default_stages
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

#### MODEL GENERATION


peak_flux_star  = nn.Parameter(torch.FloatTensor([1.]))

wf_npix = 1024
diameter = 6.603464
DEVICE = 'cuda'
num_wl = 21
oversample = 2
data_dir = '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/uuu'
num_det_px = 80

# Stellar position
offset_STAR = nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE)

# TODO: wrapper function to do all of this at once for the simpler cases, including using STPSF to generate the masks and such
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir)

# Set up the wavefront object: this will be propagated by our optical model
wavefront_broad = BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR).to(DEVICE)

diffnircam = assemble_JWST_NIRCam_coron(data_dir,num_det_px)

result = diffnircam(wavefront_broad)

result = result.detach().cpu().numpy()

np.save('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/res2.npy', result)


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

optimizer = torch.optim.AdamW(optimizer_argument, weight_decay=0.0)

# Make eta_min the same as OPDs LR, we just find it does well, but we might want something different at some point
# because making all parameters have the same LR as the minimum limit may be sub-optimal

# TODO: we might want to get rid fo the scheduler altogether at some point, or 
# test how the eta_min affects parameters with different learning rates
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_iters, eta_min=4e-9) 


######### LOAD RELEVANT DATA
refdatapath = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/NEWGOODDATA/onlycalints_nanreplaced/jw01386001001_0310e_00001_nrcalong_calints.fits'
scidatapath = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/NEWGOODDATA/onlycalints_nanreplaced/jw01386002001_0310a_00001_nrcalong_calints.fits'
ref, ref_noise = load_and_crop_data(refdatapath, num_det_px=80, coron_type='MASK335R')
sci, sci_noise = load_and_crop_data(scidatapath, num_det_px=80, coron_type='MASK335R')

# Array with 0s and 1s, used to mask certain pixels (e.g., dead or hot pixels) from contributing to the loss function
# Has to be provided by the user, probably after a first pass. Will depend on the pixel cleaning routine used.
# It can also be used to mask out full regions of the image from the optimization, for arbitrary testing.
px_mask = np.load('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/HIP65426/model_setup/pixel_masks/pixel_masks.npy')
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

loss_clip_nsigma = None # MAD sigma clipping of the science-stage loss (stage 2.2), e.g. 5.; None: plain loss

stage_cutoffs = [600, 3500, 5500, 6100, 11000] # Iterations separating stage 1.1, 1.2, 1.3, 2.1 and 2.2 in our default stage framework


tbar = tqdm.tqdm(range(scheduler_iters + 1), mininterval=30.0)

# Track injected planets
signal_peaks_arr, signal_losses_arr, snrs_arr, track_iters = [], [], [], []

for i in tbar:

    # Re start optimizer at the start always (IMPORTANT!)
    optimizer.zero_grad()

    # Model new PSF
    PSF = diffnircam(wavefront_broad)

    if i < stage_cutoffs[2]: # Iteration dividing stage 1.3 from 2.1: go from calibration on reference to fitting science
        loss = F.smooth_l1_loss(PSF, ref, reduction='none') 
    else:
        loss = F.l1_loss(PSF, sci, reduction='none') * sci_loss_weight

    weight = px_mask
    if loss_clip_nsigma is not None and i >= stage_cutoffs[3]:
        # MAD sigma clipping: pixels where the residual exceeds loss_clip_nsigma robust sigmas (1.4826 * median |residual|)
        # get zero weight this iteration, so that a planet (or anything the model cannot produce) stays out of the fit
        with torch.no_grad():
            res = (sci - PSF)[px_mask > 0]
            clip = ((sci - PSF) > loss_clip_nsigma * 1.4826 * res.abs().median()).float()
        weight = px_mask * (1. - clip)
    loss *= weight # Mask out bad pixels so they don't contribute to loss function
    loss = loss.sum() / weight.sum() # Collapse loss into a number

    # At this step, we can add some regularization to the loss function if we want. E.g., smoothness or whatever
    # Otherwise, continue to backpropagation and parameter update

    # Backpropagate loss function, populate parameter gradients
    loss.backward()

    if i < stage_cutoffs[0]: # TODO: fix flux fitting with gradient; then this will not be necessary
        wavefront_broad = hardcoded_match_flux(PSF,ref,px_mask,wavefront_broad)
    elif i > stage_cutoffs[2] and i < stage_cutoffs[3]:
        wavefront_broad = hardcoded_match_flux(PSF,sci,px_mask,wavefront_broad)

    default_stages(diffnircam, wavefront_broad, i, stage_cutoffs=stage_cutoffs) # Zero-out (in-place) gradients of parameters we are not fitting at a given stage

    # Update parameters 
    optimizer.step()
    scheduler.step()

    if i == stage_cutoffs[2] -1:
        # End of calibration!
        residual_ref = ref - PSF 
        residual_ref = residual_ref.detach().cpu().numpy()
        np.save('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/residual_ref.npy', residual_ref)

    if i == stage_cutoffs[-1]:
        # End! 
        residual_sci = sci - PSF 
        residual_sci = residual_sci.detach().cpu().numpy()
        np.save('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/residual_sci.npy', residual_sci)
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