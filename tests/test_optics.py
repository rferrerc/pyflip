"""Run from the repository root: python -m pytest tests"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'refactoring'))
import numpy as np
import torch
from scipy.special import j1, jn_zeros
from optical_elements import DiffImageOptic, BabinetImageOptic, blc_conjugate_mask
from model_classes import ShiftModule

N, M = 16, 32
WAVELENGTHS = np.array([4.0e-6, 5.0e-6])
PIXEL_SCALE = 6.6 / N
torch.manual_seed(0)
FIELD = torch.randn(len(WAVELENGTHS), N, N, dtype=torch.complex128)


def test_babinet_matches_fft_chain_without_mount():
    """stpsf's occulter through Babinet against the same occulter sampled on the FFT grid, with a
    mount piece made transparent there; the FFT grid's 2 samples per wavelength / D limit the agreement."""
    sigma = jn_zeros(1, 2)[1] / 0.3
    y, x = np.meshgrid(*(np.arange(M) - M // 2,) * 2, indexing='ij')
    fpm = []
    for wavelength in WAVELENGTHS:
        s = np.clip(sigma * np.hypot(x, y) * wavelength / (M * PIXEL_SCALE) * 206264.806, 1e-300, jn_zeros(1, 2)[1])
        fpm.append(1 - (2 * j1(s) / s)**2)
    fpm = np.stack(fpm)
    fpm[:, 2:6, 24:30] = 0.
    no_mount = fpm.copy()
    no_mount[:, 2:6, 24:30] = 1.
    babinet = BabinetImageOptic(*blc_conjugate_mask(sigma), WAVELENGTHS, PIXEL_SCALE, N, device='cpu')(FIELD, N)
    fft = DiffImageOptic(amplitude=torch.as_tensor(no_mount))(FIELD, N)
    assert (babinet - fft).norm() < 1e-2 * (fft - FIELD).norm()
    with_mount = DiffImageOptic(amplitude=torch.as_tensor(fpm))(FIELD, N)
    print('dropping the mount changes the field by %.3f (relative)' % ((with_mount - fft).norm() / fft.norm()).item())


def test_shift_module():
    mask = FIELD.abs()[:1]
    shift = ShiftModule(N, N, device='cpu')
    torch.testing.assert_close(shift(mask), mask, rtol=0, atol=1e-15)
    with torch.no_grad():
        shift.x_shift.fill_(2.)
        shift.y_shift.fill_(-1.)
    expected = torch.roll(mask, (-1, 2), (-2, -1))
    expected[..., -1:, :] = 0.
    expected[..., :, :2] = 0.
    torch.testing.assert_close(shift(mask), expected, rtol=0, atol=1e-15)
    with torch.no_grad():
        shift.x_shift.fill_(0.)
        shift.y_shift.fill_(0.)
        shift.rotation.fill_(90.)
    torch.testing.assert_close(shift(mask), torch.rot90(mask, 1, (-1, -2)), rtol=0, atol=1e-14)
