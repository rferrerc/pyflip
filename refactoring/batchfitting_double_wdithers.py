import os 
os.environ["STPSF_PATH"] = "/projects/b1094/rodrigoyeah/optics_jwst/data/stpsf-data"
os.environ["PYSYN_CDBS"] = "/projects/b1094/rodrigoyeah/optics_jwst/data/synphot-data"

from wavefronts import BroadbandWavefront
from model_building import create_STPSF_model, generate_STPSF_model_params, assemble_JWST_NIRCam_coron
from optimization_handlers import default_stages, cycling_pos_and_lyot_stages,pos_and_lyot_stages,pos_and_lyot_all,first_onlypos_then_onlyrest,looping_pos_opd_lyot,fit_separately
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

num_diths = 9

#### MODEL GENERATION

save_data_in = '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/aflepdithers_9_32_noopd'

if not os.path.isdir(save_data_in):
    os.mkdir(save_data_in)

peak_flux_star  = nn.Parameter(torch.FloatTensor([1.]))

wf_npix = 1024
diameter = 6.603464
DEVICE = 'cuda'
num_wl = 21
oversample = 2
data_dir_jan = '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/forAFLep_jan'
data_dir_oct = '/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/forAFLep_oct'
num_det_px = 80

#####

sciroll1_OCT = '/projects/b1094/rodrigoyeah/optics_jwst/data/OBS/nanreplaced_onlyfiles_AFLep/jw04558001001_03106_00001_nrcalong_calints.fits'

refdith1_OCT_files = ['/projects/b1094/rodrigoyeah/optics_jwst/data/OBS/nanreplaced_onlyfiles_AFLep/jw04558003001_03106_0000{}_nrcalong_calints.fits'.format(i+1) for i in range(num_diths)]

#2023-10-12

sciroll1_JAN = '/projects/b1094/rodrigoyeah/optics_jwst/data/OBS/nanreplaced_onlyfiles_AFLep/jw04558007001_03106_00001_nrcalong_calints.fits'

refdith1_JAN_files = ['/projects/b1094/rodrigoyeah/optics_jwst/data/OBS/nanreplaced_onlyfiles_AFLep/jw04558009001_03106_0000{}_nrcalong_calints.fits'.format(i+1) for i in range(num_diths)]
#2024-01-14

# Stellar position
offset_STAR = [nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE) for i in range(num_diths)]

######### JAN
# stpsfmodel = create_STPSF_model(from_obs=sciroll1_JAN)

# generate_STPSF_model_params(stpsfmodel,spec_type='F8V', catalog_spec='phoenix',
#                             folder_to_save=data_dir)

diffnircam_jan = assemble_JWST_NIRCam_coron(data_dir_jan,80)

# TODO: wrapper function to do all of this at once for the simpler cases, including using STPSF to generate the masks and such
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir_jan)

# Set up the wavefront object: this will be propagated by our optical model
wavefront_broad_jan_dithers = [BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR[i]).to(DEVICE) for i in range(num_diths)]

# diffnircam = assemble_JWST_NIRCam_coron(data_dir,num_det_px)

result = diffnircam_jan(wavefront_broad_jan_dithers[0])

result = result.detach().cpu().numpy()

# savingfolder = ''

np.save('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/aflep/res_jan.npy', result)

print('aaa')
print(diffnircam_jan.free_parameters.keys())



##### OCT
stpsfmodel_oct = create_STPSF_model(from_obs=sciroll1_OCT)

generate_STPSF_model_params(stpsfmodel_oct,spec_type='F8V', catalog_spec='phoenix',
                            folder_to_save=data_dir_oct)

diffnircam_oct = assemble_JWST_NIRCam_coron(data_dir_oct,80)

# TODO: wrapper function to do all of this at once for the simpler cases, including using STPSF to generate the masks and such
_, _, _, _, wlen_weights, _ = load_model_data(data_dir=data_dir_oct)


offset_STAR2 = [nn.Parameter(torch.FloatTensor([0., 0.])).to(DEVICE) for i in range(num_diths)]

# # Set up the wavefront object: this will be propagated by our optical model
wavefront_broad_oct_dithers = [BroadbandWavefront(wf_npix, diameter, wlen_weights, peak_flux_star, offset_STAR2[i]).to(DEVICE) for i in range(num_diths)]

# diffnircam = assemble_JWST_NIRCam_coron(data_dir,num_det_px)

result = diffnircam_oct(wavefront_broad_oct_dithers[0])

result = result.detach().cpu().numpy()

# savingfolder = ''

np.save('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/aflep/res_oct.npy', result)



#### OPTIMIZER SETUP

scheduler_iters = 20000 # Max number of iterations for LR scheduler: we can halt the optimization before but not after this

# Choose parameters that are going to be optimized, and assign them reasonable learning rates

#### auuu
# # OPD in priamry and post Lyot stop
# opds = [
#     # diffnircam.optical_system.JWSTPrimary.wfe_offsets.grid,
#     diffnircam.optical_system.NIRCamLyotstop.wfe_offsets.grid
# ]

# # Star position offset from coronagraph center
# star_position = [
#     wavefront_broad.angles_offset
# ]

# # Detector flats 
# detector_flats = [
#     diffnircam.optical_system.NIRCamdetector.detector_flat_field.grid
# ]

# # The 4 works well for many datasets but could be tweaked in some cases; e.g. we have found it needs to be
# # smaller for fainter stars sometimes
# optimizer_argument = [
#         {'params': opds, 'lr': 4e-9},
#         {'params': star_position, 'lr': 4e-8},
#         {'params': detector_flats, 'lr': 4e-6}
#     ]
#### au

### Set equalities!
diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.x_shift = diffnircam_oct.optical_system.NIRCamLyotstop.pupil_shifts.x_shift
diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.y_shift = diffnircam_oct.optical_system.NIRCamLyotstop.pupil_shifts.y_shift

diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.rotation = diffnircam_oct.optical_system.NIRCamLyotstop.pupil_shifts.rotation

diffnircam_jan.optical_system.NIRCamLyotstop.wfe_offsets.grid = diffnircam_oct.optical_system.NIRCamLyotstop.wfe_offsets.grid

# OPD in priamry and post Lyot stop

opds = [
    # diffnircam.optical_system.JWSTPrimary.wfe_offsets.grid,
    diffnircam_jan.optical_system.NIRCamLyotstop.wfe_offsets.grid
]

shifts = [
    diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.x_shift,
    diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.y_shift
]

rots = [
    diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.rotation
]

# Star position offset from coronagraph center
# star_position = [
#     wavefront_broad_jan.angles_offset, wavefront_broad_oct.angles_offset
# ]

star_position = [wvjan.angles_offset for wvjan in wavefront_broad_jan_dithers] + [wvoct.angles_offset for wvoct in wavefront_broad_oct_dithers]



# Detector flats 
# detector_flats = [
#     diffnircam.optical_system.NIRCamdetector.detector_flat_field.grid
# ]

# The 4 works well for many datasets but could be tweaked in some cases; e.g. we have found it needs to be
# smaller for fainter stars sometimes
optimizer_argument = [
        # {'params': opds, 'lr': 1e-9},
        {'params': star_position, 'lr': 4e-8},
        {'params': shifts, 'lr': 7e-2},
        {'params': rots, 'lr': 9e-1}
    ]


optimizer = torch.optim.AdamW(optimizer_argument, weight_decay=0.0)

# Make eta_min the same as OPDs LR, we just find it does well, but we might want something different at some point
# because making all parameters have the same LR as the minimum limit may be sub-optimal

# TODO: we might want to get rid fo the scheduler altogether at some point, or 
# test how the eta_min affects parameters with different learning rates
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_iters, eta_min=4e-9) 


######### LOAD RELEVANT DATA
# refdatapath = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/NEWGOODDATA/onlycalints_nanreplaced/jw01386001001_0310e_00001_nrcalong_calints.fits'
# scidatapath = '/projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/NEWGOODDATA/onlycalints_nanreplaced/jw01386002001_0310a_00001_nrcalong_calints.fits'
ref_jan = [] 
ref_noise_jan = []

for i in range(num_diths):
    data, noise = load_and_crop_data(refdith1_JAN_files[i], num_det_px=80, coron_type='MASK335R')
    ref_jan.append(data)
    ref_noise_jan.append(noise)


# sci, sci_noise = load_and_crop_data(sciroll1_JAN, num_det_px=80, coron_type='MASK335R')

# ref_oct, ref_noise_oct = load_and_crop_data(refdith1_OCT, num_det_px=80, coron_type='MASK335R')

ref_oct = []
ref_noise_oct = []

for i in range(num_diths):
    data, noise = load_and_crop_data(refdith1_OCT_files[i], num_det_px=80, coron_type='MASK335R')
    ref_oct.append(data)
    ref_noise_oct.append(noise)

# sci, sci_noise = load_and_crop_data(sciroll1_JAN, num_det_px=80, coron_type='MASK335R')

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

# fakes = [{'pos_x_px':10., 'pos_y_px': 0.}] # fake planet locations, relative to center of the array, in px

# sci, injected_fluxes = inject_fake_planets(sci,sci_noise, wavefront_broad, diffnircam,fakes, flux_n_5_sigma, smooth_sigma_for_flux=1.)

ref_jan = [torch.from_numpy(im).to(DEVICE) for im in ref_jan]
ref_oct = [torch.from_numpy(im).to(DEVICE) for im in ref_oct]
# sci = torch.from_numpy(sci).to(DEVICE)

####### RUN OPTIMIZATION
sci_loss_weight = 0.005 # How much to weigh down the loss function when evaluating on science target relative to reference target; essentially make optimization slower

measure_SNR_every = 300 # measure the SNR of the fake planet(s) every how many iterations 

stage_cutoffs = [600, 7000, 7500, 6100, 11000] # Iterations separating stage 1.1, 1.2, 1.3, 2.1 and 2.2 in our default stage framework


tbar = tqdm.tqdm(range(scheduler_iters + 1), mininterval=30.0)

# Track injected planets
signal_peaks_arr, signal_losses_arr, snrs_arr, track_iters = [], [], [], []

save_loss_every = 400

losses = []
loss_iters = []
lyotx1 = []
lyoty1 = []
lyotrot1 = []
poss1 = []

lyotx2 = []
lyoty2 = []
lyotrot2 = []
poss2 = []

for i in tbar:

    # Re start optimizer at the start always (IMPORTANT!)
    optimizer.zero_grad()

    # Model new PSF
    PSF_jan = [diffnircam_jan(wv) for wv in wavefront_broad_jan_dithers]
    loss = 0
    if i < stage_cutoffs[2]: # Iteration dividing stage 1.3 from 2.1: go from calibration on reference to fitting science
        for j in range(num_diths):
            loss+= F.smooth_l1_loss(PSF_jan[j], ref_jan[j], reduction='none') 
    else:
        pass
        # loss = F.l1_loss(PSF, sci, reduction='none') * sci_loss_weight

    loss *= px_mask # Mask out bad pixels so they don't contribute to loss function
    loss = loss.sum() / px_mask.sum() # Collapse loss into a number

    PSF_oct = [diffnircam_oct(wv) for wv in wavefront_broad_oct_dithers]
    loss2 = 0
    if i < stage_cutoffs[2]: # Iteration dividing stage 1.3 from 2.1: go from calibration on reference to fitting science
        for j in range(num_diths):
            loss2+= F.smooth_l1_loss(PSF_oct[j], ref_oct[j], reduction='none') 
    else:
        pass

    loss2 *= px_mask # Mask out bad pixels so they don't contribute to loss function
    loss+= loss2.sum() / px_mask.sum() # Collapse loss into a number
    
    loss /=18.
    # At this step, we can add some regularization to the loss function if we want. E.g., smoothness or whatever
    # Otherwise, continue to backpropagation and parameter update

    # Backpropagate loss function, populate parameter gradients
    loss.backward()

    if i%save_loss_every==0:
        losses.append(loss.item())
        loss_iters.append(i)
        lyotx1.append(diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.detach().cpu().numpy())
        lyoty1.append(diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.detach().cpu().numpy())
        lyotrot1.append(diffnircam_jan.optical_system.NIRCamLyotstop.pupil_shifts.rotation.detach().cpu().numpy())
        poss1.append([wavefront_broad_jan_dithers[j].angles_offset.detach().cpu().numpy() for j in range(num_diths)])

        lyotx2.append(diffnircam_oct.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.detach().cpu().numpy())
        lyoty2.append(diffnircam_oct.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.detach().cpu().numpy())
        lyotrot2.append(diffnircam_oct.optical_system.NIRCamLyotstop.pupil_shifts.rotation.detach().cpu().numpy())
        poss2.append([wavefront_broad_oct_dithers[j].angles_offset.detach().cpu().numpy() for j in range(num_diths)])

    # if i < stage_cutoffs[0]: # TODO: fix flux fitting with gradient; then this will not be necessary
    # batch = int(np.ceil(i/100))   
    # if batch%2==1: 
    wavefront_broad_jan_dithers = [hardcoded_match_flux(PSF_jan[j],ref_jan[j],px_mask,wavefront_broad_jan_dithers[j]) for j in range(num_diths)]
    wavefront_broad_oct_dithers = [hardcoded_match_flux(PSF_oct[j],ref_oct[j],px_mask,wavefront_broad_oct_dithers[j]) for j in range(num_diths)]

    # elif i > stage_cutoffs[2] and i < stage_cutoffs[3]:
    #     wavefront_broad = hardcoded_match_flux(PSF,sci,px_mask,wavefront_broad)


    # fit_separately(diffnircam_jan, wavefront_broad_jan, i, stage_cutoffs=stage_cutoffs)
    # fit_separately(diffnircam_oct, wavefront_broad_oct, i, stage_cutoffs=stage_cutoffs)
    # cycling_pos_and_lyot_stages(diffnircam_jan, wavefront_broad_jan, i, stage_cutoffs=stage_cutoffs) # Zero-out (in-place) gradients of parameters we are not fitting at a given stage
    # cycling_pos_and_lyot_stages(diffnircam_oct, wavefront_broad_oct, i, stage_cutoffs=stage_cutoffs) # Zero-out (in-place) gradients of parameters we are not fitting at a given stage

    # Update parameters 
    optimizer.step()
    scheduler.step()

    if i == stage_cutoffs[1] -1:
        # End of calibration!
        for j in range(num_diths):
            with torch.no_grad():
                residual_ref_jan = ref_jan[j] - PSF_jan[j] 
                residual_ref_jan = residual_ref_jan.detach().cpu().numpy()
                model_psf_jan = PSF_jan[j].detach().cpu().numpy()

                residual_ref_oct = ref_oct[j] - PSF_oct[j]
                residual_ref_oct = residual_ref_oct.detach().cpu().numpy()
                model_psf_oct = PSF_oct[j].detach().cpu().numpy()

                np.save(save_data_in+'/residual_ref_jan{}.npy'.format(j), residual_ref_jan)
                np.save(save_data_in+'/model_psf_ref_jan{}.npy'.format(j), model_psf_jan)

                np.save(save_data_in+'/residual_ref_oct{}.npy'.format(j), residual_ref_oct)
                np.save(save_data_in+'/model_psf_ref_oct{}.npy'.format(j), model_psf_oct)

        np.save(save_data_in+'/loss.npy', np.array(losses))
        np.save(save_data_in+'/loss_iters.npy', np.array(loss_iters))
        
        np.save(save_data_in+'/lyot_x_jan.npy', np.array(lyotx1))
        np.save(save_data_in+'/lyot_y_jan.npy', np.array(lyoty1))
        np.save(save_data_in+'/lyot_rot_jan.npy', np.array(lyotrot1))
        np.save(save_data_in+'/angle_offsets_jan.npy', np.array(poss1))
        np.save(save_data_in+'/OPD_jan.npy', diffnircam_jan.optical_system.NIRCamLyotstop.wfe_offsets.grid.detach().cpu().numpy())

        
        np.save(save_data_in+'/lyot_x_oct.npy', np.array(lyotx2))
        np.save(save_data_in+'/lyot_y_oct.npy', np.array(lyoty2))
        np.save(save_data_in+'/lyot_rot_oct.npy', np.array(lyotrot2))
        np.save(save_data_in+'/angle_offsets_oct.npy', np.array(poss2))
        np.save(save_data_in+'/OPD_oct.npy', diffnircam_oct.optical_system.NIRCamLyotstop.wfe_offsets.grid.detach().cpu().numpy())

        break
#     if i == stage_cutoffs[-1]:
#         # End! 
#         residual_sci = sci - PSF 
#         residual_sci = residual_sci.detach().cpu().numpy()
#         np.save('/projects/b1094/rodrigoyeah/optics_jwst/flip_tentative/refactoring/residual_sci.npy', residual_sci)
#         break
        


#     # Now, track SNR of injected planet(s)
#     if i%measure_SNR_every == 0 and i > stage_cutoffs[2]: # only measure in science target
#         residual_sci = sci - PSF 
#         residual_sci = residual_sci.detach().cpu().numpy()
#         signal_peaks, signal_losses, snrs = track_SNR(residual_sci, i, fakes, injected_fluxes,mask_name='MASK335R', snr_smooth_sigma=1., px_mask_numpy=px_mask_numpy)
#         signal_peaks_arr.append(signal_peaks)
#         signal_losses_arr.append(signal_losses)
#         snrs_arr.append(snrs)
#         track_iters.append(i)

# # After everything, save these diagnostics and such 
# signal_peaks_arr = np.array(signal_peaks_arr)
# signal_losses_arr = np.array(signal_losses_arr)
# snrs_arr = np.array(snrs_arr)
# track_iters = np.array(track_iters)