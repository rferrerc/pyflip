import torch.nn as nn
import torch
import numpy as np 
if __package__:
    from .model_classes import OPDOffsetModule, LearnableGaussianBlur, FlatFieldingModule, DetSubPixShift
    from .dl_utils import crop_to, partial_MFT, shift_image_subpixel
    from .diffraction import FresnelTransfer
else:
    from model_classes import OPDOffsetModule, LearnableGaussianBlur, FlatFieldingModule, DetSubPixShift
    from dl_utils import crop_to, partial_MFT, shift_image_subpixel
    from diffraction import FresnelTransfer

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
    

class DiffLyotOptic(DiffPupilOptic):
    """Lyot stop with optional axial displacement from the nominal pupil plane.

    Attributes
    ----------
    axial_offset : torch.nn.Parameter or None
        Stop displacement [m] in the equivalent optical system used for pupil
        sampling. Positive values move downstream. None uses the nominal stop.

    The OPD remains at the nominal pupil plane, after propagation back from the
    displaced stop. Only the amplitude transmission moves with the stop.
    """
    def __init__(self, name=None, opd=None, amplitude=None, wfe_offsets=None, axial_offset=None):
        super().__init__(name, opd, amplitude, wfe_offsets)
        self.axial_offset = None
        if axial_offset is not None:
            self.axial_offset = nn.Parameter(torch.as_tensor(
                axial_offset, dtype=torch.float64, device=amplitude.device))

    def forward(self, phasors, wls, normalize=False, d_pupil=None):
        """Apply the stop using Fresnel propagation to its displaced plane.

        Parameters
        ----------
        phasors : torch.Tensor
            Complex pupil field, shape (..., n_wl, n, n). Retain the full grid
            returned from the focal plane when axial displacement is enabled.
        wls : torch.Tensor
            Wavelengths of the light [m], shape (n_wl,).
        normalize : bool, optional
            Normalize the transmitted field to unit summed intensity.
        d_pupil : float or torch.Tensor, optional
            Spatial sampling interval of the pupil field [m]. Required for
            axial displacement.

        Returns
        -------
        torch.Tensor
            Field at the nominal pupil plane, cropped to the stop array size.
        """
        if self.axial_offset is None:
            return super().forward(phasors, wls, normalize)
        returned = self.displaced_stop(phasors, wls, d_pupil)
        layer = self.wfe_offsets(self.opd)
        wfe = torch.exp(1j * layer.unsqueeze(0) * 2 * np.pi / wls.unsqueeze(1).unsqueeze(2))
        new_phasor = returned * wfe
        if normalize:
            new_phasor = new_phasor / new_phasor.abs().square().sum(dim=(-2, -1), keepdim=True).sqrt()
        return new_phasor

    def displaced_stop(self, phasors, wls, d_pupil):
        """Apply the displaced stop transmission and return to the nominal pupil.

        Uses the same field and sampling arguments as forward, without applying
        an OPD. The output is cropped to the stop array size.
        """
        if d_pupil is None:
            raise ValueError('axial Lyot displacement requires d_pupil')
        n_stop = self.amplitude.shape[-1]
        n_pad = phasors.shape[-1] - n_stop
        if n_pad < 0 or n_pad % 2:
            raise ValueError('pupil field must contain the stop on a grid of matching parity')
        transmission = torch.nn.functional.pad(self.amplitude, (n_pad // 2,) * 4)
        wavelengths = wls.to(device=phasors.device, dtype=torch.float64)
        propagators = [FresnelTransfer(d_pupil, wl) for wl in wavelengths]

        def propagate(field, distance):
            return torch.stack([
                propagator.propagate(field[..., i, :, :], distance)
                for i, propagator in enumerate(propagators)
            ], dim=-3)

        displaced = propagate(phasors, self.axial_offset)
        returned = propagate(displaced * transmission, -self.axial_offset)
        return crop_to(returned, n_stop)


class DiffImageOptic(DiffOpticalElement):
    def __init__(self,name=None, amplitude=None, axial_offset=None, focal_length=None):
        super().__init__()
        self.name =name
        self.amplitude = amplitude
        self.optic_type = 'image'
        self.focal_length = focal_length
        self.axial_offset = None
        if axial_offset is not None:
            if focal_length is None or focal_length <= 0:
                raise ValueError('axial displacement requires a positive equivalent focal length')
            self.axial_offset = nn.Parameter(torch.as_tensor(
                axial_offset, dtype=torch.float64, device=amplitude.device))

    def forward(self,phasor,npixels_in,oversample=2, wavelengths=None, d_pupil=None, crop_output=True):
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
        if self.axial_offset is None:
            new_phasor = phasors * self.amplitude
        else:
            new_phasor = self.displaced_mask(phasors, wavelengths, d_pupil)

        # Back to pupil plane and format. TODO: only transform if needed
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        if crop_output:
            new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def displaced_mask(self, phasors, wavelengths, d_pupil):
        """Apply an axially displaced focal plane mask using Fresnel diffraction.

        Parameters
        ----------
        phasors : torch.Tensor
            Complex field at the nominal focal plane, shape (..., n_wl, n, n).
            Requires an even number of samples across the full focal FFT grid.
        wavelengths : torch.Tensor
            Wavelengths of the light [m], shape (n_wl,).
        d_pupil : float or torch.Tensor
            Spatial sampling interval of the pupil field [m].

        Returns
        -------
        torch.Tensor
            Field after transmission through the displaced mask and Fresnel
            propagation back to the nominal focal plane.

        Notes
        -----
        Pupil sampling, focal length and axial displacement refer to the same
        equivalent optical system. Propagation uses the transfer function
        method, including at zero displacement.
        """
        n = phasors.shape[-1]
        if n % 2:
            raise ValueError('axial masks currently require an even focal grid')
        if wavelengths is None or d_pupil is None:
            raise ValueError('axial masks require wavelengths and d_pupil')
        if phasors.shape[-3] != len(wavelengths):
            raise ValueError('the dimension before the spatial axes must index wavelength')

        # The unshifted pupil IFFT introduces a linear phase. Removing it gives the physical
        # focal field (with reversed axes), anti-periodic because the pupil samples
        # are half-integers. A half-bin frequency offset retains that boundary
        # condition without allocating a doubled focal window.
        axis = torch.arange(n, device=phasors.device, dtype=torch.float64) - n // 2
        phase_factor = torch.exp(2j * torch.pi * ((n - 1) / 2) / n *
                                 (axis[:, None] + axis[None, :])).to(phasors.dtype)
        wavelengths = wavelengths.to(device=phasors.device, dtype=torch.float64)
        focal_sampling = wavelengths * self.focal_length / (n * d_pupil)
        propagators = [
            FresnelTransfer(d_focal, wl, frequency_offset=0.5 / (n * d_focal))
            for wl, d_focal in zip(wavelengths, focal_sampling)
        ]

        def propagate(field, distance):
            return torch.stack([
                propagator.propagate(field[..., i, :, :], distance)
                for i, propagator in enumerate(propagators)
            ], dim=-3)

        displaced = propagate(phasors / phase_factor, self.axial_offset)
        returned = propagate(displaced * self.amplitude, -self.axial_offset)
        return returned * phase_factor
    

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
        self.detector_flat_field = FlatFieldingModule(self.num_det_px,self.num_det_px, device=device)

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
        for i, element in enumerate(self.optical_system):
            if element.optic_type == 'pupil':
                if isinstance(element, DiffLyotOptic):
                    phasor = element(phasor, broadbandwavefront.wavelengths, normalize=normalize,
                                     d_pupil=broadbandwavefront.pixel_scale)
                else:
                    phasor = element(phasor, broadbandwavefront.wavelengths, normalize=normalize)
                normalize = False # Only normalize at the telescope aperture, first pupil optic
            elif element.optic_type == 'image':
                next_element = self.optical_system[i + 1] if i + 1 < len(self.optical_system) else None
                # A displaced Lyot stop needs the field outside the nominal pupil.
                full_pupil = isinstance(next_element, DiffLyotOptic) and next_element.axial_offset is not None
                phasor = element(phasor,broadbandwavefront.npixels,oversample=self.oversample,
                                 wavelengths=broadbandwavefront.wavelengths,
                                 d_pupil=broadbandwavefront.pixel_scale, crop_output=not full_pupil)
            elif element.optic_type == 'detector':
                detector_image = element(phasor, broadbandwavefront)

        return detector_image
