import torch.nn as nn
import torch
import numpy as np 
from scipy.special import j1, jn_zeros
from model_classes import OPDOffsetModule, LearnableGaussianBlur,FlatFieldingModule, DetSubPixShift,ShiftModule
from dl_utils import crop_to, partial_MFT, shift_image_subpixel
from util import centred_axis, mft_matrix, mft
import sys 

class DiffOpticalElement(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.optic_type = None

    def forward(self):
        # The forward function is called by calling the class itself.
        # DiffOpticalElement(input) = DiffOpticalElement.forward(input)
        pass


class DiffPupilOptic(DiffOpticalElement):
    def __init__(self,name=None, opd=None, amplitude=None,wfe_offsets=None):
        super().__init__()
        self.name = name
        self.amplitude = amplitude
        self.opd = opd
        self.optic_type = 'pupil'
        if wfe_offsets is None: 
            self.wfe_offsets = OPDOffsetModule(opd.shape[-2], opd.shape[-1])
        else:
            self.wfe_offsets = wfe_offsets

        self.pupil_shifts = ShiftModule(amplitude.shape[-2], amplitude.shape[-1])

    def forward(self,phasors, wls, normalize=False):
        # Phasors is the result of broadbandwavefront.get_phasors().
        # Ideally, that is run at the DiffOpticalSystem level, so these only need to
        # take in the phasors and do the math.
        # TODO: be more elegant about this, if it does not compromise speed.

        # Through OPD
        layer = self.wfe_offsets(self.opd)
        wfe = torch.exp(1j * layer.unsqueeze(0) * 2 * np.pi / wls.unsqueeze(1).unsqueeze(2))
        phasors = phasors * wfe
    
        # Through amplitude transmission
        new_phasor = phasors * self.pupil_shifts(self.amplitude)
        if normalize:
            denom = new_phasor.abs() ** 2
            denom = torch.sum(denom, dim=(-2, -1), keepdim=True) ** 0.5
        else:
            denom = 1.
        return new_phasor / denom
    

class DiffImageOptic(DiffOpticalElement):
    def __init__(self,name=None, amplitude=None):
        super().__init__()
        self.name =name
        self.amplitude = amplitude
        self.optic_type = 'image'

    def forward(self,phasor,npixels_in,oversample=2):
        # Phasors is the result of broadbandwavefront.get_phasors().
        # Ideally, that is run at the DiffOpticalSystem level, so these only need to
        # take in the phasors and do the math.
        # TODO: be more elegant about this, if it does not compromise speed.

        # Propagate wavefront to image plane. TODO: only transform if needed
        if oversample > 1:
            _npixels = (npixels_in * (oversample - 1)) // 2
            phasor = torch.nn.functional.pad(phasor, (_npixels, ) * 4)

        phasors = torch.fft.fftshift(torch.fft.ifft2(phasor), dim=[-2, -1])

        # Apply transmission (e.g., focal plaen mask)
        new_phasor = phasors * self.amplitude

        # Back to pupil plane and format. TODO: only transform if needed
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor
    

BLC_SIGMA = {'MASK210R': 5.253, 'MASK335R': 3.2927866, 'MASK430R': 2.58832}


def blc_conjugate_mask(sigma):
    """stpsf's NIRCam band-limited circular occulter as a function of position on the sky:
    its conjugate mask (2 J1(sigma r) / (sigma r))^2, truncated after the first sidelobe.

    Parameters
    ----------
    sigma : float
        Profile parameter [1 / arcsec], BLC_SIGMA[image_mask].

    Returns
    -------
    conjugate_mask : callable
        1 - t at positions (y, x) [rad].
    radius : float
        Radius of the occulter [rad].
    """
    edge = jn_zeros(1, 2)[1]

    def conjugate_mask(y, x):
        s = np.clip(sigma * np.hypot(x, y) * 206264.806, np.finfo(float).tiny, edge)
        return (2 * j1(s) / s)**2

    return conjugate_mask, edge / sigma / 206264.806


class BabinetImageOptic(DiffOpticalElement):
    """Focal plane mask applied through Babinet's principle: E_out = E - T_back[m T[E]], with m
    the conjugate mask 1 - t on a window around the optical axis and T the matrix Fourier
    transform of the pupil field onto that window (util.mft). The mask is a function of
    position on the sky, sampled on the window at every wavelength.

    Parameters
    ----------
    conjugate_mask : callable
        1 - transmission at positions (y, x) [rad], arrays of one shape.
    radius : float
        Radius beyond which the mask is clear [rad].
    wavelengths : array_like
        Wavelengths [m], shape (n_wl,).
    pixel_scale : float
        Pupil sampling [m].
    npixels_in : int
        Number of pupil samples.
    samples_per_resel : float, optional
        Window samples per wavelength / D, D = npixels_in pixel_scale. The window period is
        then samples_per_resel D, beyond the 3.5 D over which the mask returns light.
    margin : float, optional
        Window margin beyond the mask [wavelength / D].

    Attributes
    ----------
    conjugate_mask : torch.Tensor
        The mask on the window, shape (n_wl, N_out, N_out).
    matrix : torch.Tensor
        Pupil to window transforms, shape (n_wl, npixels_in, N_out).
    mult : float
        Normalisation of the transform there and back, (pixel_scale d_out / wavelength)^2.
    """
    def __init__(self, conjugate_mask, radius, wavelengths, pixel_scale, npixels_in, samples_per_resel=4.5, margin=10.,
                 name=None, device='cuda'):
        super().__init__()
        self.name = name
        self.optic_type = 'image'
        D = npixels_in * pixel_scale
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        N_out = 2 * int(np.ceil(samples_per_resel * (radius * D / wavelengths.min() + margin))) + 1
        masks, matrices = [], []
        for wavelength in wavelengths:
            d_out = wavelength / (samples_per_resel * D)
            theta = centred_axis(N_out, d_out).numpy()
            masks.append(torch.as_tensor(conjugate_mask(theta[:, None], theta[None, :]), dtype=torch.float64))
            matrices.append(mft_matrix(npixels_in, pixel_scale, N_out, d_out, wavelength))
        self.register_buffer('conjugate_mask', torch.stack(masks).to(device))
        self.register_buffer('matrix', torch.stack(matrices).to(device))
        self.mult = (pixel_scale / (samples_per_resel * D))**2

    def forward(self, phasor, npixels_in, oversample=2):
        matrix = self.matrix.to(phasor.dtype)
        back = matrix.conj().transpose(-2, -1)
        focal = mft(phasor, matrix, matrix) * self.mult
        return phasor - mft(focal * self.conjugate_mask.to(phasor.dtype), back, back)


class DiffDetector(DiffOpticalElement):
    def __init__(self,args,oversample=2, name=None, num_det_px=80, shift_x=-0.12, shift_y=-0.11, device='cuda'):
        super().__init__()
        # Projects a broadbandwavefront phasor to a detetor, using the MFT algorithm
        self.name = name
        self.oversample = oversample
        self.optic_type = 'detector'

        # Unpack for MFT calculations
        (npixels, wavelengths, true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse) = args

        self.num_det_px = num_det_px#npixels

        xmats, ymats, mults = [], [], []
        for i in range(len(wavelengths)):
            args = (npixels, wavelengths[i].item(), true_pixel_scale, psf_npix*self.oversample, psf_pixel_scale/self.oversample, focal_length, shift, pixel, inverse)
            x_mat, y_mat, mult = partial_MFT(*args)
            xmats.append(x_mat.to(torch.complex64)); ymats.append(y_mat.to(torch.complex64)); mults.append(torch.tensor(mult, dtype=torch.float32))
        self.x_mat = nn.Parameter(torch.stack(xmats), requires_grad=False).to(device)
        self.y_mat = nn.Parameter(torch.stack(ymats), requires_grad=False).to(device)
        self.mult = nn.Parameter(torch.stack(mults), requires_grad=False).to(device)

        # Extra detector effects
        self.subpixel_shift = DetSubPixShift(shift_x, shift_y)

        # 0th order brighter-fatter effect
        if self.oversample!=1:
            kernel_sigma = 0.28 * self.oversample
            kernel_size = round(4.0 * kernel_sigma)

            self.charge_diffusion = LearnableGaussianBlur(kernel_size=kernel_size, init_sigma=kernel_sigma)
        else:
            self.charge_diffusion = LearnableGaussianBlur()

        # Flat-fields
        self.detector_flat_field = FlatFieldingModule(self.num_det_px,self.num_det_px)

    def forward(self,phasors, broadbandwavefront):
        
        # The phasors do the math; broadbandwavefront provides some attributes

        phasor = (self.y_mat.transpose(-2, -1) @ phasors) @ self.x_mat

        phasor *= self.mult.view(1, 1, -1, 1, 1)
        w = (broadbandwavefront.peak_flux) ** 0.5
        out = (torch.abs(phasor) * w) ** 2 

        output = out * broadbandwavefront.wl_weights.view(1, 1, -1, 1, 1)

        output = self.charge_diffusion(output)

        if self.oversample != 1:
            output = torch.sum(torch.reshape(output, (1,len(broadbandwavefront.wl_weights),self.num_det_px,self.oversample,self.num_det_px,self.oversample)), (1,3,5))

        output = broadbandwavefront.flux_correction(output)

        output = self.subpixel_shift(output)

        output = self.detector_flat_field(output)

        return output
    
class DiffOpticalSystem(nn.Module):
    def __init__(self, name=None, optical_element_list = None, oversample = 2):
        super().__init__()
        self.name = name 
        # We want parameters to be in a dictionary that preserves key order
        if sys.version_info >= (3, 7):
            # Can use normal dictionary as it preserves key order
            element_dict = {}
            for element in optical_element_list:
                element_dict[element.name.replace(" ", "")] = element
        else:
            # have to use an OrderedDict
            from collections import OrderedDict
            element_dict = OrderedDict([(element.name.replace(" ", ""),element) for element in optical_element_list])
            print("Older than Python 3.7")
            

        #self.optical_system = nn.ModuleList(optical_element_list) # Assumes first optic is a pupil plane and last is a detector; TODO: make more flexible later
        self.optical_system = nn.ModuleDict(element_dict)
        self.oversample = oversample # Can in principle be different from the detector oversampling

    @property
    def free_parameters(self):
        return dict(self.named_parameters())

    def forward(self, broadbandwavefront):
        phasor = broadbandwavefront.get_phasors()
        normalize=True
        for key in self.optical_system.keys():
            element = self.optical_system[key]
            if element.optic_type == 'pupil':
                phasor = element(phasor, broadbandwavefront.wavelengths, normalize=normalize)
                normalize = False # Only normalize at the telescope aperture, first pupil optic
            elif element.optic_type == 'image':
                phasor = element(phasor,broadbandwavefront.npixels,oversample=self.oversample)
            elif element.optic_type == 'detector':
                detector_image = element(phasor, broadbandwavefront)

        return detector_image
