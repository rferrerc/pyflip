import numpy as np
import pytest
import torch

from dl_utils import partial_MFT, pixel_coords
from optical_elements import DiffImageOptic
from reference_optics import (direct_inverse_dft, direct_mft, legacy_focal,
                              legacy_pupil, legacy_mft_matrices)
from propagation import pupil_to_focal, focal_to_pupil, mft_matrices, apply_mft


def field(n, dtype=torch.complex128):
    generator = torch.Generator().manual_seed(81)
    return torch.randn(1, n, n, generator=generator, dtype=dtype)


@pytest.mark.parametrize("n,oversample", [(6, 1), (8, 2), (8, 3)])
def test_transparent_mask_round_trip(n, oversample):
    incident = field(n)
    optic = DiffImageOptic(amplitude=torch.ones(n*oversample, n*oversample, dtype=torch.float64))
    torch.testing.assert_close(optic(incident, n, oversample), incident, rtol=1e-13, atol=1e-14)


def test_masked_round_trip_against_direct_dft():
    n, oversample = 6, 2
    incident = field(n)
    padded = torch.nn.functional.pad(incident, (3,)*4)
    focal = torch.fft.fftshift(direct_inverse_dft(padded), dim=(-2, -1))
    mask = torch.linspace(0.1, 0.9, 144, dtype=torch.float64).reshape(12, 12)
    # Forward DFT from the conjugation identity, without using an FFT oracle.
    unshifted = torch.fft.fftshift(focal*mask, dim=(-2, -1))
    returned = direct_inverse_dft(unshifted.conj()).conj()*12**2
    expected = returned[..., 3:9, 3:9]
    actual = DiffImageOptic(amplitude=mask)(incident, n, oversample)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)


def test_pupil_coordinates_have_half_pixel_origin():
    coords = pixel_coords(8, 8.)
    expected = torch.arange(8, dtype=torch.float64) - 3.5
    torch.testing.assert_close(coords[0, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(coords[1, :, 0], expected, rtol=0, atol=0)


def test_mft_against_direct_integral_and_parseval():
    incident = field(8)
    x, y, mult = partial_MFT(8, 1., 0.125, 8, 1.)
    actual = (y.T @ incident) @ x * mult
    torch.testing.assert_close(actual, direct_mft(incident, 1., 0.125, 8, 1.), rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(actual.abs().square().sum(), incident.abs().square().sum(), rtol=1e-13, atol=1e-14)


def test_positive_source_angle_moves_toward_positive_detector_x():
    # Positive-sign MFT and negative entrance phase slope must agree.
    n = 8
    xcoords = pixel_coords(n, 1.)[0]
    incident = torch.exp(-2j*np.pi*xcoords)
    x, y, mult = partial_MFT(n, 1., 1./n, 9, 1.)
    intensity = ((y.T @ incident) @ x * mult).abs().square()
    assert tuple(torch.nonzero(intensity == intensity.max())[0]) == (4, 5)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Existing inverse focal shift is fftshift instead of ifftshift on odd grids")
def test_odd_transparent_mask_round_trip():
    incident = field(9)
    actual = DiffImageOptic(amplitude=torch.ones(9, 9))(incident, 9, 1)
    torch.testing.assert_close(actual, incident, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("n,oversample", [(6, 1), (8, 2), (8, 3)])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_extracted_fft_fields_and_gradients(n, oversample, dtype):
    incident = field(n, dtype).requires_grad_(True)
    original = incident.detach().clone()
    focal = pupil_to_focal(incident, n, oversample)
    expected = legacy_focal(incident, n, oversample)
    torch.testing.assert_close(focal, expected, rtol=0, atol=0)
    mask = torch.linspace(0.1, 0.9, focal.shape[-1]**2, dtype=focal.real.dtype).reshape(focal.shape[-2:])
    actual = focal_to_pupil(focal*mask, n)
    reference = legacy_pupil(expected*mask, n)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    grad = torch.autograd.grad(actual.abs().square().sum(), incident)[0]
    ref_grad = torch.autograd.grad(reference.abs().square().sum(), incident)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
    torch.testing.assert_close(incident, original, rtol=0, atol=0)


@pytest.mark.parametrize("oversample", [1, 2, 3])
@pytest.mark.parametrize("inverse,focal_length", [(False, None), (True, 131.4)])
def test_extracted_mft_matrices(oversample, inverse, focal_length):
    args = (8, torch.tensor([3.9e-6, 4.4e-6], dtype=torch.float64), 6.603464/8,
            6, 3e-7, focal_length, [0.25, -0.125], True, inverse)
    actual = mft_matrices(*args, oversample=oversample)
    expected = legacy_mft_matrices(args, oversample)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_extracted_detector_field(optical_case):
    from reference_optics import legacy_trace
    c = optical_case
    reference = legacy_trace(c)
    x, y, mult = mft_matrices(*c.args, oversample=c.oversample)
    actual = apply_mft(reference["lyot"], x, y, mult.view(1, 1, -1, 1, 1))
    torch.testing.assert_close(actual, reference["detector_field"], rtol=0, atol=0)
