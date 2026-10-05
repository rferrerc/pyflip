"""Small optical checks: round trips, power and independent Fourier evaluation.

Run from the repository root with: python -m pytest -q
pytest.ini selects the refactored modules. No instrument files or fitting needed.
"""
import pytest
import torch

from dl_utils import dl_MFT
from optical_elements import DiffImageOptic


def asymmetric_field(npixels):
    """A smooth field with nonuniform amplitude and phase, in double precision."""
    axis = torch.linspace(-1, 1, npixels, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    amplitude = torch.exp(-2 * (x - 0.2)**2 - 3 * (y + 0.1)**2)
    phase = 0.4*x + 0.7*y + 0.3*x*y
    return amplitude * torch.exp(1j * phase)


@pytest.mark.parametrize('npixels, oversample', [
    (8, 1), (8, 2), (8, 3),
    pytest.param(9, 1, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason='Existing odd-grid return uses fftshift instead of ifftshift',
    )),
])
def test_transparent_mask_round_trip(npixels, oversample):
    """A transparent focal mask must return the original complex pupil field."""
    pupil = asymmetric_field(npixels)
    mask = torch.ones(npixels * oversample, npixels * oversample, dtype=torch.float64)
    returned = DiffImageOptic(amplitude=mask)(pupil, npixels, oversample)

    torch.testing.assert_close(returned, pupil, rtol=1e-12, atol=1e-14)


def test_mask_amplitude_sets_transmitted_power():
    """Uniform amplitude transmission t must multiply the field by t, power by t^2."""
    pupil = asymmetric_field(8)
    transmission = 0.4
    mask = torch.full((16, 16), transmission, dtype=torch.float64)
    returned = DiffImageOptic(amplitude=mask)(pupil, 8, oversample=2)

    torch.testing.assert_close(returned, transmission * pupil, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(returned.abs().square().sum(),
                               transmission**2 * pupil.abs().square().sum(),
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('focal_pixels', [8, 16])
def test_mft_round_trip_and_power(focal_pixels):
    """A full-period MFT preserves power and inverts, also with finer focal sampling."""
    pupil = asymmetric_field(8)
    wavelength = 4.4e-6  # metres
    pupil_pitch = 0.25  # metres per sample
    angular_pitch = wavelength / (focal_pixels * pupil_pitch)  # radians per sample

    focal = dl_MFT(pupil, wavelength, pupil_pitch, focal_pixels, angular_pitch)
    returned = dl_MFT(focal, wavelength, angular_pitch, 8, pupil_pitch, inverse=True)

    # FLIP's MFT amplitudes already include the sampling normalization.
    # Sum their squared magnitudes; another pixel-area factor would count it twice.
    torch.testing.assert_close(focal.abs().square().sum(), pupil.abs().square().sum(),
                               rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(returned, pupil, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('focal_pixels', [8, 16])
def test_mft_against_direct_fourier_sum(focal_pixels):
    """Matrix multiplication and direct summation must give the same complex field."""
    pupil = asymmetric_field(8)
    wavelength = 4.4e-6
    pupil_pitch = 0.25
    angular_pitch = wavelength / (focal_pixels * pupil_pitch)

    focal = dl_MFT(pupil, wavelength, pupil_pitch, focal_pixels, angular_pitch)

    # Independently evaluate the Fourier integral on the same half-pixel-centred
    # grids, using FLIP's positive exponent and discrete-amplitude normalization.
    pupil_axis = (torch.arange(8, dtype=torch.float64) - 3.5) * pupil_pitch
    y, x = torch.meshgrid(pupil_axis, pupil_axis, indexing='ij')
    angles = (torch.arange(focal_pixels, dtype=torch.float64) - (focal_pixels - 1)/2) * angular_pitch
    direct = torch.empty((focal_pixels, focal_pixels), dtype=torch.complex128)
    for row, theta_y in enumerate(angles):
        for col, theta_x in enumerate(angles):
            phase = 2 * torch.pi / wavelength * (x * theta_x + y * theta_y)
            direct[row, col] = (pupil * torch.exp(1j * phase)).sum()
    direct *= pupil_pitch * angular_pitch / wavelength

    torch.testing.assert_close(focal, direct, rtol=1e-12, atol=1e-14)
