"""Characterization of the refactored c047964 optical path before geometry changes."""
import pytest
import torch

from reference_optics import legacy_trace


def assert_same(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_asymmetric_broadband_baseline(optical_case):
    c = optical_case
    expected = legacy_trace(c)
    incident = c.wf.get_phasors()
    pupil = c.primary(incident, c.wf.wavelengths, normalize=True)
    returned = c.image_optic(pupil, c.n, c.oversample)
    lyot = c.stop(returned, c.wf.wavelengths)
    assert_same(incident, expected["incident"])
    assert_same(pupil, expected["pupil"])
    assert_same(returned, expected["returned"])
    assert_same(lyot, expected["lyot"])
    assert_same(c.detector(lyot, c.wf), expected["image"])
    image = c.system(c.wf)
    assert image.shape == (1, c.n_det, c.n_det)
    assert torch.isfinite(image).all()
    assert_same(image, expected["image"])
    torch.testing.assert_close(pupil.abs().square().sum((-2, -1)), torch.ones(1, 3, dtype=torch.float64))


def test_baseline_gradients_and_repeat_calls(optical_case):
    c = optical_case
    parameters = [c.primary.wfe_offsets.grid, c.stop.wfe_offsets.grid, c.wf.angles,
                  c.wf.peak_flux, c.wf.flux_correction.data,
                  c.detector.charge_diffusion.log_sigma, c.detector.detector_flat_field.grid]
    weights = torch.linspace(0.2, 1.1, c.n_det**2, dtype=torch.float64).reshape(1, c.n_det, c.n_det)
    reference_grads = torch.autograd.grad((legacy_trace(c)["image"]*weights).sum(), parameters)
    for _ in range(2):
        grads = torch.autograd.grad((c.system(c.wf)*weights).sum(), parameters)
        for actual, expected in zip(grads, reference_grads):
            assert torch.isfinite(actual).all()
            assert_same(actual, expected)


def test_opd_directional_derivative(optical_case):
    c = optical_case
    grid = c.primary.wfe_offsets.grid
    direction = torch.linspace(-1, 1, grid.numel(), dtype=torch.float64).reshape_as(grid)
    weights = torch.arange(c.n_det**2, dtype=torch.float64).reshape(1, c.n_det, c.n_det)
    def loss():
        return (c.system(c.wf)*weights).sum()
    analytical = (torch.autograd.grad(loss(), grid)[0]*direction).sum()
    baseline = grid.detach().clone()
    step = 1e-11
    with torch.no_grad():
        grid.copy_(baseline + step*direction)
        plus = loss()
        grid.copy_(baseline - step*direction)
        minus = loss()
        grid.copy_(baseline)
    torch.testing.assert_close(analytical, (plus-minus)/(2*step), rtol=2e-6, atol=1e-6)


def test_spectral_weights_are_intensity_weights(optical_case):
    c = optical_case
    original = c.wf.wl_weights.detach().clone()
    with torch.no_grad():
        broadband = c.system(c.wf)
        expected = torch.zeros_like(broadband)
        for i, weight in enumerate(original):
            c.wf.wl_weights.zero_()
            c.wf.wl_weights[i] = 1
            expected += weight*c.system(c.wf)
        c.wf.wl_weights.copy_(original)
    torch.testing.assert_close(broadband, expected, rtol=1e-13, atol=1e-15)


@pytest.mark.xfail(strict=True, raises=ValueError, reason="Existing oversample=1 path does not sum wavelengths before detector shifting")
def test_detector_without_oversampling(optical_case):
    import optical_elements
    c = optical_case
    detector = optical_elements.DiffDetector(c.args, oversample=1, num_det_px=c.n_det, device="cpu").double()
    image = detector(legacy_trace(c)["lyot"], c.wf)
    assert image.shape == (1, c.n_det, c.n_det)
