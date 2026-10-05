"""Axial mask displacement within the original PyFLIP propagation pipeline."""
import torch.nn as nn

from propagation_classes import PointPropagate
from refactoring.optical_elements import DiffImageOptic, DiffLyotOptic


class DisplacementPropagate(PointPropagate):
    """Extend PointPropagate with one shared axial displacement parameter.

    Parameters
    ----------
    element : {'fpm', 'lyot'}
        Optical element to displace.
    axial_offset : torch.nn.Parameter or None
        Displacement [primary-equivalent m]. None retains nominal propagation.
    focal_length : float
        Primary-equivalent focal length [m], required for FPM displacement.

    Other arguments and the forward interface are inherited from PointPropagate.
    Detector effects, OPD models and lateral stop resampling are unchanged.
    """
    def __init__(self, *args, element='fpm', axial_offset=None, focal_length=131.4, **kwargs):
        super().__init__(*args, **kwargs)
        if element not in ('fpm', 'lyot'):
            raise ValueError('element must be fpm or lyot')
        self.element = element
        self.fpm_optic = DiffImageOptic(amplitude=self.fpm, focal_length=focal_length)
        self.lyot_optic = DiffLyotOptic(amplitude=self.lyot, wfe_offsets=nn.Identity())
        optic = self.fpm_optic if element == 'fpm' else self.lyot_optic
        if axial_offset is not None:
            if element == 'fpm' and focal_length <= 0:
                raise ValueError('FPM displacement requires a positive equivalent focal length')
            optic.axial_offset = axial_offset

    def propagate_masks(self, phasors, wavefront):
        if self.fpm_optic.axial_offset is None and self.lyot_optic.axial_offset is None:
            return super().propagate_masks(phasors, wavefront)
        self.fpm_optic.amplitude = self.fpm
        full_pupil = self.lyot_optic.axial_offset is not None
        returned = self.fpm_optic(phasors, wavefront.npixels, self.oversample,
                                  wavelengths=wavefront.wavelengths, d_pupil=wavefront.pixel_scale,
                                  crop_output=not full_pupil)
        transmission = self.lyot_shifts(self.lyot)
        if full_pupil:
            self.lyot_optic.amplitude = transmission
            return self.lyot_optic.displaced_stop(returned, wavefront.wavelengths, wavefront.pixel_scale)
        return wavefront.forward(returned, transmission)
