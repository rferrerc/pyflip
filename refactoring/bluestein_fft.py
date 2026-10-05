"""Centred Fourier sums using Bluestein convolution."""
from functools import lru_cache

import torch
from scipy.fft import next_fast_len


def _chirps(n_in, n_out, n_fft, device, dtype):
    """Calculate the input/output chirps and FFT of the convolution kernel."""
    x = torch.arange(n_in, device=device, dtype=torch.float64) - (n_in - 1) / 2
    u = torch.arange(n_out, device=device, dtype=torch.float64) - (n_out - 1) / 2
    lag = torch.arange(1 - n_in, n_out, device=device, dtype=torch.float64)
    lag = lag + (n_in - n_out) / 2
    chirp_in = torch.exp(-1j * torch.pi * x.square() / n_fft).to(dtype)
    chirp_out = torch.exp(-1j * torch.pi * u.square() / n_fft).to(dtype)
    h = torch.exp(1j * torch.pi * lag.square() / n_fft).to(dtype)
    h_fft = torch.fft.fft(h, n=next_fast_len(n_in + n_out - 1))
    return chirp_in, chirp_out, h_fft


@lru_cache(maxsize=32)
def _fixed_chirps(n_in, n_out, n_fft, device, dtype):
    """Retrieve cached chirps for fixed input and output sampling."""
    return _chirps(n_in, n_out, n_fft, device, dtype)


def bluestein_fft(field, n_out, n_fft, inverse=False):
    """Compute a zoomed 2D Fourier transform using the Bluestein algorithm.

    Parameters
    ----------
    field : torch.Tensor
        Complex input field, shape (..., n_in, n_in).
    n_out : int
        Number of output samples in each dimension.
    n_fft : float or torch.Tensor
        Equivalent zero-padded length of the input signal. May be noninteger;
        a negative value reverses the transform sign. Must be nonzero.
    inverse : bool, optional
        Use a negative exponent instead of PyFLIP's positive forward exponent.

    Returns
    -------
    torch.Tensor
        Unnormalized Fourier transform, shape (..., n_out, n_out).

    Notes
    -----
    Input and output coordinates are centered on (n - 1)/2. The equivalent
    zero-padded array is not allocated. Chirps are cached for fixed sampling
    and recalculated when gradients through the sampling are required.
    """
    n_in = field.shape[-1]
    if field.shape[-2] != n_in or not field.is_complex():
        raise ValueError('field must be a square complex array')
    if n_out < 1 or float(n_fft.detach() if torch.is_tensor(n_fft) else n_fft) == 0:
        raise ValueError('n_out must be positive and n_fft nonzero')
    n_fft = n_fft if inverse else -n_fft
    if torch.is_tensor(n_fft) and n_fft.requires_grad:
        chirp_in, chirp_out, h_fft = _chirps(n_in, n_out, n_fft, field.device, field.dtype)
    else:
        chirp_in, chirp_out, h_fft = _fixed_chirps(n_in, n_out, float(n_fft), field.device, field.dtype)
    size = h_fft.numel()
    input_field = field * chirp_in[:, None] * chirp_in[None, :]
    field_fft = torch.fft.fft2(input_field, s=(size, size))
    convolution = torch.fft.ifft2(field_fft * h_fft[:, None] * h_fft[None, :])
    result = convolution[..., n_in - 1:n_in - 1 + n_out, n_in - 1:n_in - 1 + n_out]
    return result * chirp_out[:, None] * chirp_out[None, :]


def centered_fft(field, inverse=False):
    """Compute a 2D FFT with PyFLIP's coordinate and sign conventions.

    Parameters
    ----------
    field : torch.Tensor
        Complex input field, shape (..., n, n), centered on (n - 1)/2.
    inverse : bool, optional
        Use a negative exponent instead of the positive forward exponent.

    Returns
    -------
    torch.Tensor
        Unnormalized Fourier transform on the same size grid.
    """
    n = field.shape[-1]
    axis = torch.arange(n, device=field.device, dtype=torch.float64)
    centre = (n - 1) / 2
    sign = -1 if inverse else 1
    phase_factor = torch.exp(-sign * 2j * torch.pi * centre * (axis - centre / 2) / n).to(field.dtype)
    input_field = field * phase_factor[:, None] * phase_factor[None, :]
    result = torch.fft.fft2(input_field) if inverse else torch.fft.ifft2(input_field) * n**2
    return result * phase_factor[:, None] * phase_factor[None, :]
