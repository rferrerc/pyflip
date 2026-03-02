import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 
import math 
from dl_utils import pixel_coords, crop_to, partial_MFT,shift_image_subpixel
from model_classes import ShiftModule, GridOffsetModule,FlatFieldingModule,PerMirrorZernikes, OPDOffsetModule, FluxOffsetModule,AngleOffsetModule,LinearInterpOPD,PTT_OPD,LearnableGaussianBlur



class Wavefront(nn.Module):
    def __init__(self, npixels: int, diameter: float, wavelength: float, peak_flux: float, angles = None):
        super().__init__()
        self.wavelength = nn.Parameter(wavelength, requires_grad=False)
        self.pixel_scale = nn.Parameter(torch.from_numpy(np.asarray(diameter / npixels, float)), requires_grad=False)
        self.wavenumber = 2 * np.pi / self.wavelength
        self.npixels = npixels
        self.diameter = diameter
        self.peak_flux = nn.Parameter(peak_flux, requires_grad=True)
        self.coordinates = nn.Parameter(pixel_coords(self.npixels, self.diameter), requires_grad=False)
        if angles is None:
            angles = torch.zeros(2)
        self.angles = nn.Parameter(angles)
        self.reset()

    def reset(self):
        if hasattr(self, 'amplitude'):
            self.amplitude.data = torch.ones_like(self.amplitude.data) / self.npixels**1
            self.phase.data = torch.zeros_like(self.phase.data)
        else:
            self.amplitude = nn.Parameter(torch.ones((1, self.npixels, self.npixels), dtype=torch.float64) / self.npixels**1)
            self.phase = nn.Parameter(torch.zeros((1, self.npixels, self.npixels), dtype=torch.float64))

    def get_phasor(self, angles_offset=None):
        opd = self.get_tilt_opd(angles_offset)
        return self.amplitude * torch.exp(1j * (self.phase + opd))

    def get_tilt_opd(self, angles_offset=None):
        if angles_offset is not None:
            opd = -((self.angles + angles_offset)[:, None, None] * self.coordinates).sum(0)
        else:
            opd = -(self.angles[:, None, None] * self.coordinates).sum(0)

        opd = opd[None] * self.wavenumber
        return opd

    def tilt(self, angles):
        """
        Tilts the wavefront by the (x, y) angles.

        Parameters
        ----------
        angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The tilted wavefront.
        """
        coords = self.coordinates
        opd = -(angles[:, None, None] * coords).sum(0)
        opd = opd[None]
        return self.add_opd(opd)

    def add_opd(self, opd):
        self.phase.data = self.phase.data + self.wavenumber * opd

    def propagate(self, phasor = None, pad: int = 2):
        npixels = self.npixels

        if pad > 1:
            _npixels = (npixels * (pad - 1)) // 2
            phasor = torch.nn.functional.pad(phasor, (_npixels, ) * 4)

        phasor = torch.fft.fftshift(torch.fft.ifft2(phasor), dim=[-2, -1])

        return phasor

    def forward(self, phasor, layer, normalize=False):
        new_phasor = phasor * layer
        if normalize:
            denom = new_phasor.abs() ** 2
            denom = torch.sum(denom, dim=(1, 2), keepdim=True) ** 0.5
        else:
            denom = 1.
        return new_phasor / denom

    def forward_fpm(self, phasor, layer, oversample=1):
        npixels_in = self.npixels
        phasor = self.propagate(phasor, pad=oversample)
        new_phasor = phasor * layer
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def forward_wfe(self, phasor, layer, wlen):
        wfe = torch.exp(1j * layer * 2 * np.pi / wlen)
        new_phasor = phasor * wfe
        return new_phasor


class BroadbandWavefront(nn.Module):
    def __init__(self, npixels: int, diameter: float, wavelength: list, peak_flux: float, angles = None, DEVICE='cuda'):
        super().__init__()
        self.wavelengths = nn.Parameter(torch.tensor(wavelength), requires_grad=False).to(DEVICE)
        self.pixel_scale = nn.Parameter(torch.from_numpy(np.asarray(diameter / npixels, float)), requires_grad=False)
        self.wavenumbers = 2 * np.pi / self.wavelengths
        self.npixels = npixels
        self.diameter = diameter
        self.peak_flux = nn.Parameter(peak_flux, requires_grad=True)
        self.coordinates = nn.Parameter(pixel_coords(self.npixels, self.diameter), requires_grad=False)
        if angles is None:
            angles = torch.zeros(2)
        self.angles = nn.Parameter(angles)
        self.reset()

    def reset(self):
        if hasattr(self, 'amplitude'):
            self.amplitude.data = torch.ones_like(self.amplitude.data) / self.npixels**1
            self.phase.data = torch.zeros_like(self.phase.data)
        else:
            self.amplitude = nn.Parameter(torch.ones((1, self.npixels, self.npixels), dtype=torch.float64) / self.npixels**1)
            self.phase = nn.Parameter(torch.zeros((1, self.npixels, self.npixels), dtype=torch.float64))

    def get_phasors(self, angles_offset=None):
        opd = self.get_tilts_opd(angles_offset)
        return self.amplitude * torch.exp(1j * (self.phase + opd))

    def get_tilts_opd(self, angles_offset=None):
        if angles_offset is not None:
            opd = -((self.angles + angles_offset)[:, None, None] * self.coordinates).sum(0)
        else:
            opd = -(self.angles[:, None, None] * self.coordinates).sum(0)
        opd = opd[None] * self.wavenumbers.unsqueeze(1).unsqueeze(2) 
        
        return opd

    def tilt(self, angles):
        """
        Tilts the wavefront by the (x, y) angles.

        Parameters
        ----------
        angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The tilted wavefront.
        """
        coords = self.coordinates
        opd = -(angles[:, None, None] * coords).sum(0)
        opd = opd[None]
        return self.add_opd(opd)

    def add_opd(self, opd):
        self.phase.data = self.phase.data + self.wavenumbers * opd

    def propagate(self, phasor = None, pad: int = 2):
        npixels = self.npixels

        if pad > 1:
            _npixels = (npixels * (pad - 1)) // 2
            phasor = torch.nn.functional.pad(phasor, (_npixels, ) * 4)

        phasor = torch.fft.fftshift(torch.fft.ifft2(phasor), dim=[-2, -1])
        return phasor

    def forward(self, phasor, layer, normalize=False):
        new_phasor = phasor * layer
        if normalize:
            denom = new_phasor.abs() ** 2
            denom = torch.sum(denom, dim=(-2, -1), keepdim=True) ** 0.5

        else:
            denom = 1.
        return new_phasor / denom

    def forward_fpm(self, phasors, layer, oversample=1):
        npixels_in = self.npixels
        phasors = self.propagate(phasors, pad=oversample) # up to here it's fine!
        new_phasor = phasors * layer
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def forward_wfes(self, phasors, layer, wlens):
        wfe = torch.exp(1j * layer.unsqueeze(0) * 2 * np.pi / wlens.unsqueeze(1).unsqueeze(2))
        new_phasor = phasors * wfe
        return new_phasor

class PointPropagate(nn.Module):
    def __init__(self, aperture, lyot, fpm, nircam_opd, args,oversample=None, use_ptt = None,OTE_wfe_basis=None,second_delta_wfe=None, num_det_px = 80):
        super().__init__()
        self.aperture = aperture
        self.lyot = lyot
        self.fpm = fpm
        self.nircam_opd = nircam_opd
        self.num_det_px = num_det_px
        if oversample is None:
            oversample = 1
        self.oversample = oversample

        self.lyot_shifts = ShiftModule(lyot.shape[-2], lyot.shape[-1])
        self.fpm_shifts = ShiftModule(fpm.shape[-2], fpm.shape[-1])
        
        self.nircam_offsets = GridOffsetModule(nircam_opd.shape[-2], nircam_opd.shape[-1])

        self.flux_correction = FluxOffsetModule()

        self.detector_flat_field = FlatFieldingModule(num_det_px,num_det_px)

        if self.oversample!=1:
            kernel_sigma = 0.28 * self.oversample
            kernel_size = round(4.0 * kernel_sigma)

            self.charge_diffusion = LearnableGaussianBlur(kernel_size=kernel_size, init_sigma=kernel_sigma)
        else:
            self.charge_diffusion = LearnableGaussianBlur()
        if OTE_wfe_basis is None:
            self.wfe_offsets = OPDOffsetModule(nircam_opd.shape[-2], nircam_opd.shape[-1])
        else:
            self.wfe_offsets = OTE_wfe_basis
        self.second_wfe_offsets = second_delta_wfe
        self.angle_offsets = AngleOffsetModule()

        (npixels, wavelengths, true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse) = args
        xmats, ymats, mults = [], [], []
        for i in range(len(wavelengths)):
            args = (npixels, wavelengths[i].item(), true_pixel_scale, psf_npix*self.oversample, psf_pixel_scale/self.oversample, focal_length, shift, pixel, inverse)
            x_mat, y_mat, mult = partial_MFT(*args)
            xmats.append(x_mat); ymats.append(y_mat); mults.append(torch.tensor(mult, dtype=torch.float64))
        self.x_mat = nn.Parameter(torch.stack(xmats), requires_grad=False)
        self.y_mat = nn.Parameter(torch.stack(ymats), requires_grad=False)
        self.mult = nn.Parameter(torch.stack(mults), requires_grad=False)

    # making changes here
    def forward(self, broadbandwavefront, wfe,wavelengths, wl_weights):
        output = None
        wfe_ = self.wfe_offsets(wfe)
        if self.second_wfe_offsets is not None:
            wfe_ = self.second_wfe_offsets(wfe_)

        phasors = broadbandwavefront.get_phasors(self.angle_offsets())
        phasors = broadbandwavefront.forward_wfes(phasors, wfe_, broadbandwavefront.wavelengths)
        phasors_ap = broadbandwavefront.forward(phasors, self.aperture, normalize=True)
        phasors_fpm = broadbandwavefront.forward_fpm(phasors_ap, self.fpm,oversample=self.oversample)

        phasors_lyot = broadbandwavefront.forward(phasors_fpm, self.lyot_shifts(self.lyot))
        phasors_nircam_opd = broadbandwavefront.forward_wfes(phasors_lyot, self.nircam_offsets(self.nircam_opd), broadbandwavefront.wavelengths)
        phasor = (self.y_mat.transpose(-2, -1) @ phasors_nircam_opd) @ self.x_mat

        phasor *= self.mult.view(1, 1, -1, 1, 1)
        w = (broadbandwavefront.peak_flux) ** 0.5
        out = (torch.abs(phasor) * w) ** 2 

        output = out * wl_weights.view(1, 1, -1, 1, 1)

        output = self.charge_diffusion(output)

        if self.oversample != 1:
            output = torch.sum(torch.reshape(output, (1,len(wl_weights),self.num_det_px,self.oversample,self.num_det_px,self.oversample)), (1,3,5))



        output = self.flux_correction(output)
        output = shift_image_subpixel(output)

        output = self.detector_flat_field(output)

        return output

    def forward_val(self, wavefront):
        phasor = wavefront.get_phasor()
        phasor = wavefront.forward(phasor, self.aperture)
        phasor = (self.y_mat.T @ phasor) @ self.x_mat
        phasor *= self.mult
        w = wavefront.peak_flux ** 0.5
        out = (torch.abs(phasor) * w) ** 2

        return out