import torch.nn as nn
import torch
import numpy as np 
from model_classes import OPDOffsetModule, LearnableGaussianBlur,FlatFieldingModule, DetSubPixShift
from dl_utils import crop_to, partial_MFT, shift_image_subpixel

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
        new_phasor = phasors * self.amplitude
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
            xmats.append(x_mat); ymats.append(y_mat); mults.append(torch.tensor(mult, dtype=torch.float64))
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
        self.optical_system = nn.ModuleList(optical_element_list) # Assumes first optic is a pupil plane and last is a detector; TODO: make more flexible later
        self.oversample = oversample # Can in principle be different from the detector oversampling

    @property
    def free_parameters(self):
        return dict(self.named_parameters())

    def forward(self, broadbandwavefront):
        phasor = broadbandwavefront.get_phasors()
        normalize=True
        for element in self.optical_system:
            if element.optic_type == 'pupil':
                phasor = element(phasor, broadbandwavefront.wavelengths, normalize=normalize)
                normalize = False # Only normalize at the telescope aperture, first pupil optic
            elif element.optic_type == 'image':
                phasor = element(phasor,broadbandwavefront.npixels,oversample=self.oversample)
            elif element.optic_type == 'detector':
                detector_image = element(phasor, broadbandwavefront)

        return detector_image
