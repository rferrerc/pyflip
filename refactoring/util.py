"""Fourier transforms between sampled planes as matrices."""
import numpy as np
import torch


def centred_axis(N, d, offset=0., device=None):
    """Sample positions (j - (N - 1) / 2 - offset) d, shape (N,), float64."""
    return (torch.arange(N, dtype=torch.float64, device=device) - (N - 1) / 2 - offset) * d


def mft_matrix(N_in, d_in, N_out, d_out, wavelength=1., offset_in=0., offset_out=0., device=None):
    """exp(2 pi i x_in x_out^T / wavelength) between two sampled axes, shape (N_in, N_out), complex128.

    Applied as mft(field, matrix_y, matrix_x) it transforms a field sampled at d_in onto a
    plane sampled at d_out, up to the factor d_in^2 / wavelength; the conjugate matrices
    transform back, up to d_out^2 / wavelength.

    Parameters
    ----------
    N_in, N_out : int
        Samples of the two axes.
    d_in, d_out : float
        Their sampling intervals; d_in d_out / wavelength is the spacing in the transform.
    wavelength : float, optional
        1 for a plain Fourier pair.
    offset_in, offset_out : float, optional
        Shift of each axis from its centre [samples].
    """
    x_in = centred_axis(N_in, d_in, offset_in, device)
    x_out = centred_axis(N_out, d_out, offset_out, device)
    return torch.exp(2j * np.pi * torch.outer(x_in, x_out) / wavelength)


def mft(field, matrix_y, matrix_x):
    """matrix_y^T field matrix_x over the last two axes of field."""
    return matrix_y.transpose(-2, -1) @ field @ matrix_x

