import torch.nn as nn
import torch
import numpy as np 
from dl_utils import pixel_coords
from model_classes import FluxOffsetModule

class BroadbandWavefront(nn.Module):
    def __init__(self, npixels: int, diameter: float, wavelength: list, peak_flux: float, angles = None, angles_offset=None, DEVICE='cuda'):
        super().__init__()
        self.wavelengths = nn.Parameter(torch.tensor(wavelength[0]), requires_grad=False).to(DEVICE)
        self.wl_weights = nn.Parameter(torch.tensor(wavelength[1]), requires_grad=False).to(DEVICE)
        self.pixel_scale = nn.Parameter(torch.from_numpy(np.asarray(diameter / npixels, float)), requires_grad=False)
        self.wavenumbers = 2 * np.pi / self.wavelengths
        self.npixels = npixels
        self.diameter = diameter
        self.peak_flux = nn.Parameter(peak_flux, requires_grad=True)

        self.flux_correction = FluxOffsetModule() # TODO: check if this is better than tuning peak flux above

        self.coordinates = nn.Parameter(pixel_coords(self.npixels, self.diameter), requires_grad=False)
        if angles is None:
            angles = torch.zeros(2)
        self.angles = nn.Parameter(angles).to(DEVICE)
        if angles_offset is None:
            angles_offset = torch.zeros(2)
        self.angles_offset =angles_offset.to(DEVICE) # Now angles_offset is a property of the wavefront: needs to be on the DEVICE! # TODO: look into this
        self.reset()

    def reset(self):
        if hasattr(self, 'amplitude'):
            self.amplitude.data = torch.ones_like(self.amplitude.data) / self.npixels**1
            self.phase.data = torch.zeros_like(self.phase.data)
        else:
            self.amplitude = nn.Parameter(torch.ones((1, self.npixels, self.npixels), dtype=torch.float64) / self.npixels**1)
            self.phase = nn.Parameter(torch.zeros((1, self.npixels, self.npixels), dtype=torch.float64))

    def get_phasors(self):
        opd = self.get_tilts_opd()
        return self.amplitude * torch.exp(1j * (self.phase + opd))

    def get_tilts_opd(self):
        if self.angles_offset is not None:
            opd = -((self.angles + self.angles_offset)[:, None, None] * self.coordinates).sum(0)
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