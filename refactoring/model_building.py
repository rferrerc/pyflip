import torch
from dl_utils import load_data,arcsec2rad
import numpy as np 
import torch.nn.functional as F
import torch.nn as nn
from optical_elements import DiffPupilOptic, DiffImageOptic, DiffDetector, DiffOpticalSystem

def assemble_JWST_NIRCam_coron(data_dir, num_wl, oversample,num_det_px,psf_pixel_scale=0.062424185,wf_npix=1024,diameter = 6.603464, DEVICE='cuda'):
    # Set up physical model for the observation being optimized on: load mask designs, known aberrations, etc.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psf_npix = num_det_px
    psf_pixel_scale = psf_pixel_scale # arcsec/pixel, default value in LW channel

    if oversample is None and num_wl is None:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=data_dir)
    else:
        aperture, lyot, fpm, nircam_OPD, wlen_weights = load_data(data_dir=data_dir, num_wl=num_wl, oversample=oversample)
        
    aperture = torch.FloatTensor(aperture.copy()).to(DEVICE)[None]

    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    wlen_weights = torch.FloatTensor(wlen_weights).to(DEVICE)

    sampledWFEs = np.load(f'{data_dir}/opds_dates/observation_opd.npy')
    sampledWFEs = np.flip(sampledWFEs, axis=0)[None]
    sampledWFEs = torch.from_numpy(sampledWFEs.copy()).float()
    sampledWFEs = F.interpolate(sampledWFEs[:, None], size=(wf_npix, wf_npix), mode='bilinear').squeeze()
    entrance_OPD = sampledWFEs.contiguous().to(DEVICE)
    
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

    # Now, build each element in the JWST/NIRCam optical train
    # 1. JWST primary: transmission is JWST mirrors, OPD is STPSF-processed WFE measurement.
    # 2. NIRCam FPM: Focal plane mask. Wavelength-dependent pixel-scale coronagraph transmission. Image plane
    # 3. NIRCam Lyot: Transmission is Lyot stop, OPD is NIRCam wavelength-dependent OPD 
    # 4. NIRCam detector: Projects onto detector with MFT, and applies shifts, BFE, flat fields, etc.

    JWST_primary = DiffPupilOptic(name='JWST Primary',opd=entrance_OPD, amplitude=aperture)
    NIRCam_FPM = DiffImageOptic(name='NIRCam coron focal plane mask',amplitude=fpm)
    NIRCam_Lyot = DiffPupilOptic(name='NIRCam Lyot stop',opd=nircam_OPD, amplitude=lyot)
    NIRCam_detector = DiffDetector(prop_args,name='NIRCam detector')

    JWST_NIRCam = DiffOpticalSystem(name='JWST NIRCam coron',optical_element_list=[JWST_primary, NIRCam_FPM, NIRCam_Lyot, NIRCam_detector])

    return JWST_NIRCam
 