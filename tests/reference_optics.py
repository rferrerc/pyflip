"""Test references, with no imports from the implementation being tested.

The legacy_* functions freeze the arithmetic at c047964 for extraction regression.
They intentionally retain its conventions and defects, and are not physical oracles.
The direct DFT functions provide separate small-grid mathematical checks.
"""
import numpy as np
import torch
import torch.nn.functional as F


def legacy_coords(n, scale=1.0, offset=0.0):
    # Preserve the original default-dtype rounding before conversion to float64.
    pix = torch.arange(n) - (n - 1) / 2.0
    pix *= scale
    pix -= offset
    return torch.from_numpy(np.array(np.meshgrid(pix, indexing="xy"))).double().squeeze()


def legacy_partial_mft(n, wavelength, d_in, m, d_out, focal_length=None,
                       shift=(0.0, 0.0), pixel=True, inverse=False):
    if not pixel:
        shift = np.asarray(shift) / d_out
    fringe = wavelength / (d_in*n)
    def matrix(s):
        scale = d_out / fringe
        if focal_length is not None:
            scale /= focal_length
        a = legacy_coords(n, 1.0/n, s/n)
        b = legacy_coords(m, scale, s*scale)
        phase = 2j*np.pi*torch.outer(a, b)
        if inverse:
            phase *= -1
        return torch.exp(phase)
    output_size = m*d_out
    if focal_length is not None:
        output_size /= focal_length
    mult = np.exp(np.log(output_size/fringe) - (np.log(n) + np.log(m)))
    return matrix(shift[0]), matrix(shift[1]), mult


def legacy_mft_matrices(args, oversample):
    n, wavelengths, d_in, m, d_out, focal_length, shift, pixel, inverse = args
    xs, ys, mults = [], [], []
    for wl in wavelengths:
        x, y, mult = legacy_partial_mft(n, wl.item(), d_in, m*oversample,
                                      d_out/oversample, focal_length, shift, pixel, inverse)
        xs.append(x)
        ys.append(y)
        mults.append(torch.tensor(mult, dtype=torch.float64))
    return torch.stack(xs), torch.stack(ys), torch.stack(mults)


def legacy_focal(field, n, oversample):
    if oversample > 1:
        padding = n*(oversample - 1)//2
        field = F.pad(field, (padding,)*4)
    return torch.fft.fftshift(torch.fft.ifft2(field), dim=(-2, -1))


def legacy_pupil(field, n):
    field = torch.fft.fft2(torch.fft.fftshift(field, dim=(-2, -1)))
    m = field.shape[-1]
    return field[..., (m-n)//2:(m+n)//2, (m-n)//2:(m+n)//2]


def legacy_detector(field, case):
    d, wf = case.detector, case.wf
    x, y, mult = legacy_mft_matrices(case.args, case.oversample)
    field = (y.transpose(-2, -1) @ field) @ x
    field *= mult.view(1, 1, -1, 1, 1)
    intensity = (torch.abs(field) * wf.peak_flux**0.5)**2
    intensity = intensity * wf.wl_weights.view(1, 1, -1, 1, 1)
    sigma = torch.exp(d.charge_diffusion.log_sigma)
    size = d.charge_diffusion.kernel_size
    coords = torch.arange(size, dtype=torch.double) - size//2
    kernel = torch.exp(-0.5*(coords/sigma)**2)
    kernel = kernel / kernel.sum()
    kernel = torch.outer(kernel, kernel)[None, None]
    b, c, nw, h, w = intensity.shape
    intensity = F.conv2d(intensity.view(b*nw, c, h, w), kernel, padding="same").view(b, c, nw, h, w)
    if case.oversample != 1:
        intensity = intensity.reshape(1, nw, case.n_det, case.oversample, case.n_det, case.oversample).sum((1, 3, 5))
    intensity = wf.flux_correction.data * intensity
    if intensity.dim() == 3:
        intensity = intensity.unsqueeze(0)
    b, c, h, w = intensity.shape
    theta = torch.tensor([[[1, 0, -2*d.subpixel_shift.shift_x/w],
                           [0, 1, -2*d.subpixel_shift.shift_y/h]]], dtype=torch.double).expand(b, -1, -1)
    grid = F.affine_grid(theta, intensity.size(), align_corners=False)
    intensity = F.grid_sample(intensity, grid, mode="bicubic", padding_mode="border", align_corners=False)
    if intensity.size(0) == 1:
        intensity = intensity.squeeze(0)
    return field, d.detector_flat_field.grid * intensity


def legacy_trace(case):
    wf = case.wf
    tilt = -((wf.angles + wf.angles_offset)[:, None, None] * wf.coordinates).sum(0)
    incident = wf.amplitude * torch.exp(1j*(wf.phase + tilt[None]*wf.wavenumbers[:, None, None]))
    p1 = case.primary
    phase = torch.exp(1j*(p1.opd + p1.wfe_offsets.grid).unsqueeze(0)*2*np.pi / wf.wavelengths[:, None, None])
    pupil = incident*phase*p1.amplitude
    pupil = pupil / torch.sum(pupil.abs()**2, dim=(-2, -1), keepdim=True)**0.5
    focal = legacy_focal(pupil, case.n, case.oversample)
    masked = focal*case.image_optic.amplitude
    returned = legacy_pupil(masked, case.n)
    p2 = case.stop
    phase = torch.exp(1j*(p2.opd + p2.wfe_offsets.grid).unsqueeze(0)*2*np.pi / wf.wavelengths[:, None, None])
    lyot = returned*phase*p2.amplitude
    detector_field, image = legacy_detector(lyot, case)
    return dict(incident=incident, pupil=pupil, focal=focal, masked=masked,
                returned=returned, lyot=lyot, detector_field=detector_field, image=image)


def direct_inverse_dft(field):
    """Unshifted inverse DFT from its definition, for tiny square grids."""
    n = field.shape[-1]
    j = torch.arange(n, dtype=torch.float64, device=field.device)
    matrix = torch.exp(2j*np.pi*torch.outer(j, j)/n)
    return (matrix @ field @ matrix.T) / n**2


def direct_mft(field, wavelength, d_in, m, d_out):
    """Centred positive-sign Fourier integral, with sample-area normalization.

    No production coordinate or transfer-matrix functions; double precision
    throughout. Use binary-exact sampling in strict comparisons to the legacy
    coordinate builder, which rounds its intermediate coordinates to float32.
    """
    n = field.shape[-1]
    x = (torch.arange(n, dtype=torch.float64) - (n-1)/2)*d_in
    a = (torch.arange(m, dtype=torch.float64) - (m-1)/2)*d_out
    phase = torch.exp(2j*np.pi*torch.outer(x, a)/wavelength)
    return (phase.T @ field @ phase) * (d_in*d_out/wavelength)
