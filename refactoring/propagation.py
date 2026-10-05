"""
Numerical propagation functions used by the refactored optical elements.

Sampling and wavelengths are supplied explicitly. The optical elements apply
transmission and OPD; these functions perform the Fourier transforms. Arithmetic,
centering and normalization are retained from the original implementation.
"""
import torch
import numpy as np

def calc_nfringes(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: int,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    focal_shift: float = 0.0,
):
    """
    Calculates the number of fringes in the output plane.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.

    Returns
    -------
    nfringes : Array
        The number of fringes in the output plane.
    """
    # Fringe size
    diameter = npixels_in * pixel_scale_in
    fringe_size = wavelength / diameter

    # Output array size
    output_size = npixels_out * pixel_scale_out
    if focal_length is not None:
        output_size /= focal_length + focal_shift

    # Fringe size and number of fringes
    return output_size / fringe_size

def nd_coords(npixels, pixel_scales = 1.0, offsets=0.0, indexing: str = "xy",):

    if not isinstance(npixels, tuple):
        npixels = (npixels,)
    if not isinstance(pixel_scales, tuple):
        pixel_scales = (pixel_scales,) * len(npixels)
    if not isinstance(offsets, tuple):
        offsets = (offsets,) * len(npixels)

    def pixel_fn(n, offset, scale):
        pix = torch.arange(n) - (n - 1) / 2.0
        pix *= scale
        pix -= offset
        return pix

    lin_pixels = []
    for n, o, p in zip(npixels, offsets, pixel_scales):
      lin_pixels += [pixel_fn(n, o, p)]

    positions = torch.from_numpy(np.array(np.meshgrid(*lin_pixels, indexing=indexing)))
    positions = positions.to(dtype=torch.float64)

    return torch.squeeze(positions)

def transfer_matrix(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    shift: float = 0.0,
    focal_length: float = None,
    focal_shift: float = 0.0,
    inverse: bool = False,
):
    """
    Calculates the transfer matrix for the MFT.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    shift : float = 0.0
        The shift in the center of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.
    inverse: bool = False
        Is this a forward or inverse propagation.

    Returns
    -------
    transfer_matrix : Array
        The transfer matrix for the MFT.
    """
    # Get parameters
    fringe_size = wavelength / (pixel_scale_in * npixels_in)

    # Input coordinates
    scale_in = 1.0 / npixels_in
    in_vec = nd_coords(npixels_in, scale_in, shift * scale_in)

    # Output coordinates
    scale_out = pixel_scale_out / fringe_size
    if focal_length is not None:
        # scale_out /= focal_length
        scale_out /= focal_length + focal_shift
    out_vec = nd_coords(npixels_out, scale_out, shift * scale_out)

    # Generate transfer matrix
    matrix = 2j * np.pi * torch.outer(in_vec, out_vec)
    if inverse:
        matrix *= -1

    return torch.exp(matrix)

def partial_MFT(
    npixels_in,
    wavelength: float,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    shift = [0.0, 0.0],
    pixel: bool = True,
    inverse: bool = False,
):
    if not pixel:
        shift /= pixel_scale_out

    get_tf_mat = lambda s: transfer_matrix(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        s,
        focal_length,
        0.0,
        inverse,
    )
    x_mat = get_tf_mat(shift[0])
    y_mat = get_tf_mat(shift[1])
    nfringes = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    mult = np.exp(np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out)))

    return x_mat, y_mat, mult

def crop_to(array, npixels: int):
    npixels_in = array.shape[-1]
    start, stop = (npixels_in - npixels) // 2, (npixels_in + npixels) // 2
    return array[..., start:stop, start:stop]


def pupil_to_focal(phasor, npixels_in, oversample=2):
    """
    Propagate a pupil field to the full FFT focal grid.

    Parameters
    ----------
    phasor : torch.Tensor
        Complex field with spatial dimensions on the last two axes.
    npixels_in : int
        Number of pupil samples along each axis.
    oversample : int
        Zero-padding factor. The focal angular pitch is wavelength divided by
        the padded grid size and the pupil sample pitch.

    Returns
    -------
    torch.Tensor
        Field on the focal grid, with the origin at the central FFT bin.
    """
    if oversample > 1:
        _npixels = (npixels_in * (oversample - 1)) // 2
        phasor = torch.nn.functional.pad(phasor, (_npixels, ) * 4)
    return torch.fft.fftshift(torch.fft.ifft2(phasor), dim=[-2, -1])


def focal_to_pupil(phasor, npixels_out):
    """
    Return a focal field to the pupil and crop to npixels_out samples per axis.

    Retains the original fftshift convention, which is an inverse shift only
    for even grids. Odd-grid round trips are a known, separately tested defect.
    """
    phasor = torch.fft.fft2(torch.fft.fftshift(phasor, dim=[-2, -1]))
    return crop_to(phasor, npixels_out)


def mft_matrices(npixels, wavelengths, true_pixel_scale, psf_npix, psf_pixel_scale,
                 focal_length, shift, pixel, inverse, oversample=2):
    """
    Build detector MFT matrices and normalization factors for each wavelength.

    Arguments follow partial_MFT; wavelengths is a one-dimensional tensor.
    Detector size and pitch are native pixel values. Oversampling increases
    the size and reduces the pitch by the same factor. Returns stacked CPU
    tensors (x matrices, y matrices, factors), as in the original detector.
    """
    xmats, ymats, mults = [], [], []
    for i in range(len(wavelengths)):
        args = (npixels, wavelengths[i].item(), true_pixel_scale, psf_npix*oversample,
                psf_pixel_scale/oversample, focal_length, shift, pixel, inverse)
        x_mat, y_mat, mult = partial_MFT(*args)
        xmats.append(x_mat)
        ymats.append(y_mat)
        mults.append(torch.tensor(mult, dtype=torch.float64))
    return torch.stack(xmats), torch.stack(ymats), torch.stack(mults)


def apply_mft(phasor, x_mat, y_mat, mult):
    """
    Apply precomputed MFT matrices to the last two field axes.

    Leading dimensions broadcast over wavelengths. mult is a scalar or a
    tensor shaped to broadcast over the output field. No input is modified.
    """
    phasor = (y_mat.transpose(-2, -1) @ phasor) @ x_mat
    phasor *= mult
    return phasor
