"""Small optical checks: round trips, power and independent Fourier evaluation.

Run from the repository root with: python -m pytest -q
pytest.ini selects the refactored modules. No instrument files or fitting needed.
"""
import pytest
import torch
from types import SimpleNamespace

from dl_utils import dl_MFT, crop_to
from optical_elements import DiffImageOptic, DiffPupilOptic, DiffLyotOptic, DiffOpticalSystem, DiffDetector
from bluestein_fft import bluestein_fft, centered_fft
from diffraction import FresnelSingle, FresnelTransfer


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
    d_pupil = 0.25  # metres per sample
    d_angle = wavelength / (focal_pixels * d_pupil)  # radians per sample

    focal = dl_MFT(pupil, wavelength, d_pupil, focal_pixels, d_angle)
    returned = dl_MFT(focal, wavelength, d_angle, 8, d_pupil, inverse=True)

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
    d_pupil = 0.25
    d_angle = wavelength / (focal_pixels * d_pupil)

    focal = dl_MFT(pupil, wavelength, d_pupil, focal_pixels, d_angle)

    # Independently evaluate the Fourier integral on the same half-pixel-centred
    # grids, using FLIP's positive exponent and discrete-amplitude normalization.
    pupil_axis = (torch.arange(8, dtype=torch.float64) - 3.5) * d_pupil
    y, x = torch.meshgrid(pupil_axis, pupil_axis, indexing='ij')
    angles = (torch.arange(focal_pixels, dtype=torch.float64) - (focal_pixels - 1)/2) * d_angle
    direct = torch.empty((focal_pixels, focal_pixels), dtype=torch.complex128)
    for row, theta_y in enumerate(angles):
        for col, theta_x in enumerate(angles):
            phase = 2 * torch.pi / wavelength * (x * theta_x + y * theta_y)
            direct[row, col] = (pupil * torch.exp(1j * phase)).sum()
    direct *= d_pupil * d_angle / wavelength

    torch.testing.assert_close(focal, direct, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('n_in, n_out, period', [(8, 8, 8.), (9, 9, 9.),
                                              (8, 7, 103.5), (7, 10, 17.25)])
@pytest.mark.parametrize('inverse', [False, True])
def test_bluestein_matches_pyflip_mft(n_in, n_out, period, inverse):
    """Match PyFLIP's origin, sign and amplitudes, including mixed grid parity."""
    field = asymmetric_field(n_in)
    # For wavelength=d_pupil=1, d_angle=1/period. dl_MFT includes
    # the amplitude factor 1/period, whereas the numerical Fourier sum does not.
    expected = dl_MFT(field, 1., 1., n_out, 1 / period, inverse=inverse)
    actual = bluestein_fft(field, n_out, period, inverse=inverse) / period
    # dl_MFT builds some coordinates in float32 before converting to float64.
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    x = torch.arange(n_in, dtype=torch.float64) - (n_in - 1)/2
    u = torch.arange(n_out, dtype=torch.float64) - (n_out - 1)/2
    sign = -1 if inverse else 1
    matrix = torch.exp(sign * 2j * torch.pi * u[:, None] * x[None, :] / period)
    direct = (matrix @ field @ matrix.T) / period
    torch.testing.assert_close(actual, direct, rtol=1e-12, atol=1e-14)
    if n_in == n_out == period:
        torch.testing.assert_close(centered_fft(field, inverse), actual * period,
                                   rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('z', [-0.03, 0.03])
def test_fresnel_against_direct_integral(z):
    """Bluestein evaluates the Fresnel quadrature with its physical normalization."""
    field = asymmetric_field(8)
    wavelength, d_in, d_out = 1e-6, 2e-5, 3e-5
    x = (torch.arange(8, dtype=torch.float64) - 3.5) * d_in
    u = (torch.arange(7, dtype=torch.float64) - 3) * d_out
    y, x = torch.meshgrid(x, x, indexing='ij')
    direct = torch.empty((7, 7), dtype=torch.complex128)
    for row, v in enumerate(u):
        for col, w in enumerate(u):
            kernel = torch.exp(1j * torch.pi / (wavelength * z) * ((x-w)**2 + (y-v)**2))
            direct[row, col] = (field * kernel).sum() * d_in**2 / (1j * wavelength * z)
    actual = FresnelSingle(d_in, wavelength, d_out, 7).propagate(field, z)
    torch.testing.assert_close(actual, direct, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('n', [8, 9])
@pytest.mark.parametrize('z', [-0.03, 0.03])
def test_single_fresnel_methods_round_trip_and_power(n, z):
    """FFT and Bluestein agree; a natural-grid round trip preserves physical power."""
    field = asymmetric_field(n)
    wavelength, d_in = 1e-6, 2e-5
    d_out = wavelength * abs(z) / (n * d_in)
    fft = FresnelSingle(d_in, wavelength, d_out, transform='fft').propagate(field, z)
    blue = FresnelSingle(d_in, wavelength, d_out, transform='bluestein').propagate(field, z)
    returned = FresnelSingle(d_out, wavelength, d_in, transform='bluestein').propagate(blue, -z)
    torch.testing.assert_close(fft, blue, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(returned, field, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(blue.abs().square().sum() * d_out**2,
                               field.abs().square().sum() * d_in**2, rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize('z', [-0.01, 0., 0.01])
def test_transfer_round_trip_and_distance_gradient(z):
    """Signed propagation inverts, and autodiff remains valid through focus."""
    field = asymmetric_field(9)
    distance = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    diffraction = FresnelTransfer(2e-5, 1e-6)
    propagate = lambda dz: diffraction.propagate(field, dz)
    propagated = propagate(distance)
    returned = diffraction.propagate(propagated, -distance)
    torch.testing.assert_close(returned, field, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(propagated.abs().square().sum(), field.abs().square().sum())
    assert torch.autograd.gradcheck(propagate, (distance,), eps=1e-7, atol=1e-5)


def test_bluestein_distance_gradient_and_batches():
    """Changing distance must differentiate the Fourier kernel as well as the chirps."""
    field = torch.stack([asymmetric_field(7), 2j * asymmetric_field(7)])
    z = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    diffraction = FresnelSingle(2e-5, 1e-6, 3e-5, 8)
    propagate = lambda dz: diffraction.propagate(field, dz)
    assert torch.autograd.gradcheck(propagate, (z,), eps=1e-7, atol=1e-5)
    output = propagate(z)
    torch.testing.assert_close(output[1], 2j * output[0])


def test_fresnel_formulations_agree_for_resolved_gaussian():
    """Independent Fresnel formulations agree when the field fits the sampled window."""
    n, d_x, wavelength = 128, 1e-5, 1e-6
    axis = (torch.arange(n, dtype=torch.float64) - (n - 1)/2) * d_x
    radius2 = axis[:, None].square() + axis[None, :].square()
    field = torch.exp(-radius2 / (8e-5)**2).to(torch.complex128)
    z = n * d_x**2 / wavelength
    transfer = FresnelTransfer(d_x, wavelength).propagate(field, z)
    single = FresnelSingle(d_x, wavelength, d_x).propagate(field, z)
    torch.testing.assert_close(single, transfer, rtol=1e-7, atol=1e-9)
    width2 = (8e-5)**2 + 1j * wavelength * z / torch.pi
    analytic = (8e-5)**2 / width2 * torch.exp(-radius2 / width2)
    torch.testing.assert_close(single, analytic, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('direction', [-1, 1])
def test_displaced_gaussian_mask_against_bluestein_and_analytic_field(direction):
    """A displaced Gaussian transmission gives the same field with both methods."""
    n, d_x, wavelength = 128, 1e-5, 1e-6
    axis = (torch.arange(n, dtype=torch.float64) - (n - 1)/2) * d_x
    radius2 = axis[:, None].square() + axis[None, :].square()
    beam_width, mask_width = 8e-5, 1.1e-4
    field = torch.exp(-radius2 / beam_width**2).to(torch.complex128)
    mask = torch.exp(-radius2 / mask_width**2)
    # This distance gives both formulations the same well-resolved output grid.
    z = direction * n * d_x**2 / wavelength

    transfer_method = FresnelTransfer(d_x, wavelength)
    single_method = FresnelSingle(d_x, wavelength, d_x, transform='bluestein')
    displaced = transfer_method.propagate(field, z)
    transfer = transfer_method.propagate(displaced * mask, -z)
    displaced = single_method.propagate(field, z)
    bluestein = single_method.propagate(displaced * mask, -z)

    # Gaussian propagation and multiplication have a closed-form solution.
    # The complex squared width increases by i*wavelength*z/pi in free space.
    width_at_mask = beam_width**2 + 1j * wavelength * z / torch.pi
    width_after_mask = 1 / (1 / width_at_mask + 1 / mask_width**2)
    width_returned = width_after_mask - 1j * wavelength * z / torch.pi
    amplitude = (beam_width**2 / width_at_mask) * (width_after_mask / width_returned)
    analytic = amplitude * torch.exp(-radius2 / width_returned)

    torch.testing.assert_close(transfer, analytic, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(bluestein, analytic, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(transfer, bluestein, rtol=1e-7, atol=1e-9)
    assert torch.linalg.vector_norm(analytic - field * mask) > 0.01 * torch.linalg.vector_norm(analytic)


def test_transfer_at_zero_distance_preserves_field():
    field = asymmetric_field(8)
    propagator = FresnelTransfer(2e-5, 1e-6)
    torch.testing.assert_close(propagator.propagate(field, 0.), field)


@pytest.mark.parametrize('z', [-0.002, 0., 0.002])
def test_axial_mask_against_full_period_pupil_solution(z):
    """Full-period Fresnel propagation also has an exact pupil-space solution.

    This checks the focal phase ramp and half-bin frequencies independently of
    the new propagator. The quadratic factors here are an exact DFT identity
    for this full-period test, not a substitute model for a cropped focal field.
    """
    n, oversample, d_x, focal_length = 8, 2, 0.01, 1.
    wavelengths = torch.tensor([4e-6, 5e-6], dtype=torch.float64)
    field = torch.stack([asymmetric_field(n), 1j * asymmetric_field(n)])[None]
    mask = asymmetric_field(n * oversample).abs()[None, None]
    optic = DiffImageOptic(amplitude=mask, axial_offset=z, focal_length=focal_length)
    actual = optic(field, n, oversample, wavelengths=wavelengths, d_pupil=d_x)

    axis = (torch.arange(n, dtype=torch.float64) - (n - 1)/2) * d_x
    radius2 = axis[:, None].square() + axis[None, :].square()
    phase = torch.exp(-1j * torch.pi * z * radius2 / (wavelengths[:, None, None] * focal_length**2))
    nominal = DiffImageOptic(amplitude=mask)
    expected = nominal(field * phase, n, oversample) * phase.conj()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('z', [0., 0.002])
def test_axial_mask_distance_gradient(z):
    """A nonuniform mask has a differentiable displacement even at nominal focus."""
    field = asymmetric_field(8)[None]
    mask = asymmetric_field(16).abs()
    wavelengths = torch.tensor([4e-6], dtype=torch.float64)
    optic = DiffImageOptic(amplitude=mask, axial_offset=z, focal_length=1.)

    def measurement():
        output = optic(field, 8, wavelengths=wavelengths, d_pupil=0.01)
        return output[0, 2, 5].abs().square()

    derivative, = torch.autograd.grad(measurement(), optic.axial_offset)
    step = 1e-7
    with torch.no_grad():
        optic.axial_offset.fill_(z + step)
        plus = measurement()
        optic.axial_offset.fill_(z - step)
        minus = measurement()
    finite_difference = (plus - minus) / (2 * step)
    assert abs(derivative) > 1e-5
    torch.testing.assert_close(derivative, finite_difference, rtol=1e-6, atol=1e-8)


def test_displaced_transparent_mask_round_trip():
    field = asymmetric_field(8)[None]
    optic = DiffImageOptic(amplitude=torch.ones(16, 16), axial_offset=0.002, focal_length=1.)
    returned = optic(field, 8, wavelengths=torch.tensor([4e-6], dtype=torch.float64),
                     d_pupil=0.01)
    torch.testing.assert_close(returned, field, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('direction', [-1, 1])
def test_displaced_lyot_against_bluestein(direction):
    """A displaced pupil stop agrees with independent single-transform propagation."""
    n, n_stop, d_x = 128, 64, 1e-5
    wavelengths = torch.tensor([1e-6, 1.2e-6], dtype=torch.float64)
    axis = (torch.arange(n, dtype=torch.float64) - (n - 1)/2) * d_x
    radius2 = axis[:, None].square() + axis[None, :].square()
    field = torch.exp(-radius2 / (8e-5)**2).to(torch.complex128)
    field = torch.stack([field, 1j * field])[None]
    mask = crop_to(torch.exp(-radius2 / (6e-5)**2), n_stop)
    opd = asymmetric_field(n_stop).real * 1e-8
    z = direction * n * d_x**2 / wavelengths[0]
    stop = DiffLyotOptic(opd=opd, amplitude=mask, wfe_offsets=torch.nn.Identity(),
                        axial_offset=z)
    actual = stop(field, wavelengths, d_pupil=d_x)

    transmission = torch.nn.functional.pad(mask, ((n - n_stop)//2,) * 4)
    reference = []
    for i, wavelength in enumerate(wavelengths):
        diffraction = FresnelSingle(d_x, wavelength, d_x, transform='bluestein')
        displaced = diffraction.propagate(field[..., i, :, :], z)
        returned = diffraction.propagate(displaced * transmission, -z)
        reference.append(crop_to(returned, n_stop) * torch.exp(2j * torch.pi * opd / wavelength))
    expected = torch.stack(reference, dim=-3)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('offset', [None, 0.])
def test_lyot_nominal_and_zero_displacement(offset):
    """Disabling displacement or setting it to zero recovers the nominal pupil optic."""
    field = asymmetric_field(16)[None]
    mask = asymmetric_field(8).abs()
    opd = asymmetric_field(8).real * 1e-8
    wavelengths = torch.tensor([1e-6], dtype=torch.float64)
    nominal = DiffPupilOptic(opd=opd, amplitude=mask, wfe_offsets=torch.nn.Identity())
    stop = DiffLyotOptic(opd=opd, amplitude=mask, wfe_offsets=torch.nn.Identity(),
                        axial_offset=offset)
    expected = nominal(crop_to(field, 8), wavelengths)
    input_field = crop_to(field, 8) if offset is None else field
    actual = stop(input_field, wavelengths, d_pupil=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize('z', [0., -0.001, 0.001])
def test_lyot_displacement_gradient(z):
    """The Lyot displacement remains differentiable at and on either side of zero."""
    field = asymmetric_field(16)[None]
    wavelengths = torch.tensor([1e-6], dtype=torch.float64)
    stop = DiffLyotOptic(opd=torch.zeros(8, 8), amplitude=asymmetric_field(8).abs(),
                        wfe_offsets=torch.nn.Identity(), axial_offset=z)

    def measurement():
        output = stop(field, wavelengths, d_pupil=1e-5)
        return output[0, 2, 5].abs().square()

    derivative, = torch.autograd.grad(measurement(), stop.axial_offset)
    step = 1e-8
    with torch.no_grad():
        stop.axial_offset.fill_(z + step)
        plus = measurement()
        stop.axial_offset.fill_(z - step)
        minus = measurement()
    assert abs(derivative) > 1e-5
    torch.testing.assert_close(derivative, (plus - minus) / (2 * step), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('fpm_offset', [None, 1e-4])
def test_system_retains_full_pupil_for_displaced_lyot(fpm_offset):
    """The system must pass the uncropped FPM output to a displaced Lyot stop."""
    class PupilReadout(torch.nn.Module):
        optic_type = 'detector'

        def forward(self, field, wavefront):
            return field

    field = asymmetric_field(8)[None]
    wavelengths = torch.tensor([1e-6], dtype=torch.float64)
    wavefront = SimpleNamespace(get_phasors=lambda: field, wavelengths=wavelengths,
                                npixels=8, pixel_scale=1e-5)
    primary = DiffPupilOptic(opd=torch.zeros(8, 8), amplitude=torch.ones(8, 8),
                            wfe_offsets=torch.nn.Identity())
    fpm = DiffImageOptic(amplitude=asymmetric_field(16).abs(),
                        axial_offset=fpm_offset, focal_length=0.001)
    stop = DiffLyotOptic(opd=torch.zeros(8, 8), amplitude=asymmetric_field(8).abs(),
                        wfe_offsets=torch.nn.Identity(), axial_offset=0.001)
    system = DiffOpticalSystem(optical_element_list=[primary, fpm, stop, PupilReadout()])
    received = []
    hook = stop.register_forward_pre_hook(lambda module, args: received.append(args[0]))
    output = system(wavefront)
    hook.remove()
    assert received[0].shape[-2:] == (16, 16)
    assert output.shape[-2:] == (8, 8)

    # Removing the outer field before propagation must change this case.
    truncated = torch.nn.functional.pad(crop_to(received[0], 8), (4,) * 4)
    cropped_early = stop(truncated, wavelengths, d_pupil=1e-5)
    assert torch.linalg.vector_norm(output - cropped_early) > 1e-4
    derivative, = torch.autograd.grad(output.abs().square().sum(), stop.axial_offset)
    assert torch.isfinite(derivative) and abs(derivative) > 1e-5


@pytest.mark.parametrize('element, true_offset', [('fpm', 3e-5), ('lyot', -3e-4)])
def test_optimizer_recovers_injected_offset(element, true_offset):
    """An optimizer recovers a known displacement from a detector image."""
    from wavefronts import BroadbandWavefront

    n, d_x, wavelength = 16, 1e-5, 1e-6
    wavelengths = torch.tensor([wavelength], dtype=torch.float64)
    field = asymmetric_field(n)
    primary = DiffPupilOptic(opd=field.angle()[None] * wavelength / (2 * torch.pi),
                            amplitude=field.abs()[None], wfe_offsets=torch.nn.Identity())
    fpm = DiffImageOptic(amplitude=(1 - 0.8 * asymmetric_field(2*n).abs())[None, None],
                        axial_offset=true_offset if element == 'fpm' else None, focal_length=0.001)
    lyot = DiffLyotOptic(opd=torch.zeros(1, 1, n, n), amplitude=field.abs()[None],
                        wfe_offsets=torch.nn.Identity(),
                        axial_offset=true_offset if element == 'lyot' else None)
    args = (n, wavelengths, d_x, n, wavelength / (n*d_x), None, [0., 0.], True, False)
    detector = DiffDetector(args, oversample=2, num_det_px=n, device='cpu')
    model = DiffOpticalSystem(optical_element_list=[primary, fpm, lyot, detector])
    wavefront = BroadbandWavefront(n, n*d_x, [[wavelength], [1.]],
                                  torch.ones(1, dtype=torch.float64), DEVICE='cpu')
    offset = (fpm if element == 'fpm' else lyot).axial_offset
    with torch.no_grad():
        target = model(wavefront).squeeze()
        offset.zero_()
    model.requires_grad_(False)
    wavefront.requires_grad_(False)
    offset.requires_grad_(True)
    optimizer = torch.optim.AdamW([offset], lr=abs(true_offset)/10, weight_decay=0.)
    target = target / target.sum()
    losses = []
    for _ in range(180):
        optimizer.zero_grad()
        prediction = model(wavefront).squeeze()
        loss = ((prediction / prediction.sum() - target)**2).sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 1e-5
    assert offset.item() == pytest.approx(true_offset, rel=0.01)
