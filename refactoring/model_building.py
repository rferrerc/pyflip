import torch
from dl_utils import load_model_data,arcsec2rad
import numpy as np 
import torch.nn.functional as F
import torch.nn as nn
from optical_elements import DiffPupilOptic, DiffImageOptic, DiffDetector, DiffOpticalSystem
import astropy.units as u 
import os 
import json

# Important: before importing STPSF and POPPY and using the functions in this module, 
# at least two environment variables below must be set to wherever 
# the STPSF data and PYSYNPHOT data are (see installation instructions)
# these are 'STPSF_PATH' and 'PYSYN_CDBS'. If these are not set globally,
# set them on the script from which this is being imported BEFORE importing this script. E.g.,
# os.environ["STPSF_PATH"] = "/path-to/data/stpsf-data"
# os.environ["PYSYN_CDBS"] = "/path-to/data/synphot-data"

import stpsf 
import poppy 

def create_STPSF_model(from_obs=None, instrument='NIRCam', inst_params={'filter':'F444W', 'image_mask':'MASK335R', 'pupil_mask':'CIRCLYOT', 'bar_offset':'narrow'},
                        obs_time='2022-07-29T23:00:00',choice='closest'):
    """
    Create a STPSF instrument given a combination of filters and such.
    Mostly a small convenience function to apply Kellen's pupil shear.

    If 'from_obs' is used, it will use STPSF's automatic generator from a FITS file for an observation. 
    Otherwise, parameters must be plugged in manually.

    Examples of valid/realistic parameter combinations:
    - Filter = any, image_mask = 'MASK335R', pupil_mask = 'CIRCLYOT' (for most typical round coronagraph obs)
    - Filter = any, image_mask = 'MASKLWB', pupil_mask = 'WEDGELYOT', bar_offset = 'narrow' (for bar coronagraphy in the NARROW fiducial point override)
    """
    if from_obs is not None:
        inst_instance = stpsf.setup_sim_to_match_file(from_obs)
    else: 
        if instrument == 'NIRCam':
            inst_instance = stpsf.NIRCam()
            inst_instance.filter = inst_params['filter']
            inst_instance.image_mask = inst_params['image_mask']
            inst_instance.pupil_mask = inst_params['pupil_mask']
            if inst_instance.image_mask == 'MASKLWB':
                # bar_offset keyword only makes sense in the bar coronagraph, otherwise ignore
                inst_instance.options['bar_offset'] = inst_params['bar_offset']
            opd_filename = stpsf.mast_wss.get_opd_at_time(obs_time, choice=choice) # OPD to use for the observation
            inst_instance.load_wss_opd(opd_filename, backout_si_wfe=True)  
        else:
            raise ValueError('Only NIRCam is implemented for now')
    
    # If we have a NIRCam round coron observation, use Kellen parameters to improve PSF fidelity: correct values ONLY in STPSF v2.2.0 and above
    # (i.e., after the NIRCam coronagraph OPDs were flipped for higher fidelity; see https://github.com/spacetelescope/stpsf/pull/125)
    if inst_instance.channel == 'long' and inst_instance.pupil_mask in ['CIRCLYOT','MASKRND']:
        inst_instance.options['pupil_shift_x'] = -0.01346839
        inst_instance.options['pupil_shift_y'] = -0.00762122
        inst_instance.options['pupil_rotation'] = -0.56725767
        print('AAAAAA')
        # nircam_instance.options['defocus_waves'] = -0.00162708# TODO: implement this in FLIP
    elif inst_instance.channel == 'short' and inst_instance.pupil_mask in ['CIRCLYOT','MASKRND']:
        inst_instance.options['pupil_shift_x'] =0.00898595
        inst_instance.options['pupil_shift_y'] = 0.00132024
        inst_instance.options['pupil_rotation'] = -0.08309628
        # nircam_instance.options['defocus_waves'] = -0.00883023# TODO: implement this in FLIP
    
    return inst_instance 


def generate_STPSF_model_params(stpsf_instrument, spec_type='A1V', catalog_spec='phoenix', input_spectrum=None, 
                                num_wl=21,oversample=2, folder_to_save='mydata',return_data=False):
    """
    Take a STPSF instrument and create FLIP-friendly model parameters.

    TODO: Now it is pretty hard coded to the NIRCam coronagraphs/the things we care about;
    in the future we might want to make this more flexible to turn any STPSF model automatically into FLIP format. 

    In its current form, this saves the following instrument parameters formatted for FLIP:
    - The JWST primary mirror transmission  
    - The JWST primary mirror OPD 
    - The image plane focal plane mask PER WAVELENGTH, because currently we follow STPSF, and each wavelength results
        in a different pixel scale when FFT'd into the image plane. It also has the indicated oversampling (we typically use 2048 pixels).
    - The Lyot stop transmission
    - The combined NIRCam OPD PER WAVELENGTH, because it has chromatic components.
        This is the COMBINED OPD effect, and it incorporates the OPD at of the NIRCam coronagraph itself at this field point,
        wavelength-dependent defocus, and wavelength-dependent OPD from the coronagraph wedge.
        In the future these could be separated if we want to isolate them; currently we combine them (following STPSF)

    As a rule of thumb, we typically use 21 wavelengths for wide filters and 9 wavelengths for medium filters,
    but this has not been rigorously optimized. 
    """
    npix=1024 * int(oversample) # Image plane pixels: this is oversampling of 2. Pupil plane is always in 1024x1024 pixels for simplicity
    
    # Set up NIRCam instrument
    if not os.path.isdir(folder_to_save):
        os.mkdir(folder_to_save)
    folder_oversample_and_wl_num = os.path.join(folder_to_save, f'wavelength_dependents')
    if not os.path.isdir(folder_oversample_and_wl_num):
        os.mkdir(folder_oversample_and_wl_num)

    # Set up spectral type of star
    if input_spectrum is not None:
        # provided by user; anything that STPSF/POPPY can take as the "source" keyword in calc_psf
        src_spectrum = input_spectrum 
    else:
        # Synthesize a spectrum from a spectral type and catalog
        # Currently, this requires that the key matches *exactly* the catalog name and will raise an error if no perfect match exists, 
        # which is good (unlike webbpsf_ext that may just ignore your keyword) and bad (because the match must be exact and not all possible
        # spectral types are in all catalogs). This can be circumvented by providing your own nice spectrum above,
        # or by using a "close enough" spectral type to your source.
        src_spectrum = stpsf.specFromSpectralType(spec_type, catalog=catalog_spec) 

    # Get OTE OPD data
    optsys = stpsf_instrument.get_optical_system(fft_oversample=int(npix/1024))

    opd_observation = optsys.planes[0].opd

    opd_folder = os.path.join(folder_to_save, 'opds')
    aperture_folder = os.path.join(folder_to_save, 'pupil_masks')

    if not os.path.isdir(opd_folder):
        os.mkdir(opd_folder)
    
    if not os.path.isdir(aperture_folder):
        os.mkdir(aperture_folder)

    opd_filepath = os.path.join(opd_folder, 'observation_opd.npy')

    np.save(opd_filepath, opd_observation)

    psf_nircam_instance = stpsf_instrument.calc_psf(oversample=int(npix/1024),source=src_spectrum, nlambda=num_wl)

    # Get wavelengths and their weights
    wlens = []
    wlens_weights = []
    for attribute in psf_nircam_instance[0].header:
        if attribute.startswith('WAVE') and not attribute.endswith('N'):
            wlens.append(psf_nircam_instance[0].header[attribute])
        elif attribute.startswith('WGHT'):
            wlens_weights.append(psf_nircam_instance[0].header[attribute])

    wlen_wlen_we_arr = np.array([wlens, wlens_weights])
    wlen_wlen_we_arr

    wavelengths_folder = os.path.join(folder_oversample_and_wl_num, 'wlens_weights')

    # Below, axis 0 is wavelength in micron, axis 1 is weight (for a weighted-sum when combining many monochromatic wavefronts into a broad-band one)
    wlenweights_path = os.path.join(wavelengths_folder, 'lambda_weights.npy') 

    # Get wavelength dependent focal plane masks and NIRCam OPDs
    masks_transmissions = []
    masks_filenames = []
    nircamopds = []
    nircamopds_filenames = []

    transmissions_folder = os.path.join(folder_oversample_and_wl_num,'image_mask_transmissions')
    nircamopds_folder = os.path.join(folder_oversample_and_wl_num,'fov_wl_nircam_opds')

    for w in range(len(wlens)):
        wf1 = poppy.Wavefront(wavelength=wlens[w] * u.meter,diam=6.603464 * u.meter, oversample=int(npix/1024))
        wf1.propagate_to(optsys.planes[2])
        arr = optsys.planes[2].get_transmission(wf1)
        masks_transmissions.append(arr)
        masks_filenames.append(os.path.join(transmissions_folder, 'image_mask_transmission_{}.npy'.format(w)))
        totopd = optsys.planes[4].get_opd(wlens[w])
        nircamopds.append(totopd)
        nircamopds_filenames.append(os.path.join(nircamopds_folder, 'fov_wl_nircam_opd_{}.npy'.format(w)))


    # Save everything
    if not os.path.isdir(wavelengths_folder):
        os.mkdir(wavelengths_folder)
    np.save(wlenweights_path, wlen_wlen_we_arr)

    if not os.path.isdir(opd_folder):
        os.mkdir(opd_folder)

    np.save(os.path.join(aperture_folder, 'lyotstop_transmission.npy'), optsys[3].amplitude)
    np.save(os.path.join(aperture_folder, 'primary_transmission.npy'), optsys[0].amplitude)

    if not os.path.isdir(transmissions_folder):
        os.mkdir(transmissions_folder)

    if not os.path.isdir(nircamopds_folder):
        os.mkdir(nircamopds_folder)

    for i in range(len(wlens)):
        np.save(masks_filenames[i],masks_transmissions[i])
        np.save(nircamopds_filenames[i],nircamopds[i])

    # Log information
    prop_dict = {}

    prop_dict['instrument'] = stpsf_instrument.name
    prop_dict['filter'] = stpsf_instrument.filter
    prop_dict['image_mask'] = stpsf_instrument.image_mask
    prop_dict['pupil_mask'] = stpsf_instrument.pupil_mask
    prop_dict['pupil_shift_x'] = stpsf_instrument.options['pupil_shift_x']
    prop_dict['pupil_shift_y'] = stpsf_instrument.options['pupil_shift_y']
    prop_dict['pupil_rotation'] = stpsf_instrument.options['pupil_rotation']
    prop_dict['img_oversample'] = oversample
    prop_dict['num_wl'] = num_wl 

    if 'bar_offset' in stpsf_instrument.options.keys():
        prop_dict['bar_offset'] = stpsf_instrument.options['bar_offset'] 

    with open(os.path.join(folder_to_save, 'instrument_config.json'), "w") as f:
        json.dump(prop_dict, f, indent=4)

    if return_data:
        return wlen_wlen_we_arr, opd_observation,masks_transmissions,nircamopds, optsys,psf_nircam_instance
    else:
        return None

def assemble_JWST_NIRCam_coron(data_dir,num_det_px,psf_pixel_scale=0.062424185,diameter = 6.603464, DEVICE='cuda'):
    """
    Assemble a FLIP NIRCam coron model, given some model parameters.
    Again, quite hard coded:
    - data_dir must be where the data is, and the folder structure must follow what 'generate_STPSF_model_params' generates.
        This argument must be the same as folder_to_save in the function above. 
    - psf_pixel_scale: Must match the STPSF model: 0.062424185 if channel == 'long'; half of that if 'short'.
        TODO: Clean this up to get pixel scale automtically from STPSF insturment, maybe with a wrapper 
    """
    
    # Set up physical model for the observation being optimized on: load mask designs, known aberrations, etc.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wf_npix = 1024
    psf_npix = num_det_px
    psf_pixel_scale = psf_pixel_scale # arcsec/pixel, default value in LW channel

    aperture, lyot, fpm, nircam_OPD, wlen_weights, sampledWFEs = load_model_data(data_dir)

    aperture = torch.FloatTensor(aperture.copy()).to(DEVICE)[None]

    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float32).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    wlen_weights = torch.FloatTensor(wlen_weights).to(DEVICE)

    sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
    sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    entrance_OPD = sampledWFEs.contiguous().to(DEVICE)
    
    # Set up the propagation model parameters
    shift = [0.0, 0.0]
    pixel = True
    focal_length = None
    npixels = 1024
    inverse = False
    true_pixel_scale = diameter / npixels
    psf_pixel_scale_arcsec = psf_pixel_scale
    psf_pixel_scale = arcsec2rad(psf_pixel_scale)
    prop_args = (npixels, wlen_weights[0], true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse)

    # Now, build each element in the JWST/NIRCam optical train
    # 1. JWST primary: transmission is JWST mirrors, OPD is STPSF-processed WFE measurement.
    # 2. NIRCam FPM: Focal plane mask. Wavelength-dependent pixel-scale coronagraph transmission. Image plane
    # 3. NIRCam Lyot: Transmission is Lyot stop, OPD is NIRCam wavelength-dependent OPD 
    # 4. NIRCam detector: Projects onto detector with MFT, and applies shifts, BFE, flat fields, etc.

    JWST_primary = DiffPupilOptic(name='JWSTPrimary',opd=entrance_OPD, amplitude=aperture)
    NIRCam_FPM = DiffImageOptic(name='NIRCam coron focal plane mask',amplitude=fpm)
    NIRCam_Lyot = DiffPupilOptic(name='NIRCamLyotstop',opd=nircam_OPD, amplitude=lyot)

    # Big TODO:
    # Now, our detector model applies a sub-pixel shift to align the center of the coronagraph in the 
    # PSF model with the center of the coronagraph in the image data. The integer pixel shift is already
    # "applied" by the dl_utils.load_and_crop_data function, which crops the data so that the center of
    # the coronagraph is as close as possible to the center of the Npix x Npix array, up to an integer number
    # of pixels. We don't want to interpolate our data to avoid artifacts, so the last "sub-pixel" shifting/interpolation
    # must be done in our model.

    # This sub-pixel shift is coronagraph and filter dependent; so we read-in the instrument configuration file 
    # to know what instrument/filter we are using, and perform the corresponding sub-pixel shift, hoping that the
    # integer-pixel shift (cropping the data) was done correctly.

    # Right now this is hardcoded and only for a couple of filters, so ideally in the future we make both this routine and the cropping-the-data
    # routine more flexible e.g. with Jarron's shifts. We should also test this more rigorously at some point,
    # since now it is ignoring e.g. the filter differences in the MASKLWB_NARROW configuration.

    # It is not terribly important for detection, but we definitely will need to refine this for accurate astrometry.

    with open(os.path.join(data_dir, 'instrument_config.json'), "r") as f:
        inst_properties = json.load(f)

    subpix_x, subpix_y =  0.,0.

    if inst_properties['image_mask'] == 'MASK335R' and inst_properties['filter'] == 'F444W':
        subpix_x, subpix_y = -0.536,-0.526 # -0.12,-0.11 plus the -0.416 px shift the former 2x2 charge diffusion kernel used to apply
    elif inst_properties['image_mask'] == 'MASKLWB':
        subpix_x, subpix_y = -0.086,-0.526 # Based on F410M if I remember correctly, but should double check (0.33,-0.11 plus the -0.416 px of the former 2x2 kernel)

    NIRCam_detector = DiffDetector(prop_args,shift_x=subpix_x, shift_y=subpix_y, name='NIRCamdetector')

    JWST_NIRCam = DiffOpticalSystem(name='JWSTNIRCamcoron',optical_element_list=[JWST_primary, NIRCam_FPM, NIRCam_Lyot, NIRCam_detector])

    return JWST_NIRCam
 