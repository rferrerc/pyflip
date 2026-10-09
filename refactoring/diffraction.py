"""Fresnel diffraction on square, uniformly sampled grids.

Coordinates are centered on (n - 1)/2. Integrated intensity is calculated as
sum(abs(field)**2) * d**2, where d is the spatial sampling interval [m].
Both formulations omit the common propagation phase exp(i k z).
"""
import math

import torch

if __package__:
    from .bluestein_fft import bluestein_fft, centered_fft
else:
    from bluestein_fft import bluestein_fft, centered_fft


def propagate_wavelengths(phasors, propagators, distance):
    """Propagate each wavelength using its corresponding propagator.

    Parameters
    ----------
    phasors : torch.Tensor
        Complex fields, shape (..., n_wl, n, n).
    propagators : sequence of Fresnel
        One propagator per wavelength, in the same order as phasors.
    distance : float or torch.Tensor
        Propagation distance [m].

    Returns
    -------
    torch.Tensor
        Propagated fields with wavelength on the third-to-last axis.
    """
    return torch.stack([
        propagator.propagate(phasors[..., i, :, :], distance)
        for i, propagator in enumerate(propagators)
    ], dim=-3)


class Fresnel:
    """Common sampling and wavelength for Fresnel diffraction.

    Attributes
    ----------
    d_in : float or torch.Tensor
        Spatial sampling interval of the input field [m].
    wavelength : float or torch.Tensor
        Wavelength of the light [m].
    """
    def __init__(self, d_in, wavelength):
        self.d_in = d_in
        self.wavelength = wavelength


class FresnelTransfer(Fresnel):
    """Fresnel diffraction using the transfer function method.

    Input and output have the same spatial sampling and number of samples.
    For an isolated field, the computational window must be large enough
    to avoid overlap between periodic copies.

    Attributes
    ----------
    frequency_offset : float or torch.Tensor
        Frequency sampling offset in both dimensions [m^-1]. Zero describes
        a periodic field. Half a frequency sample describes a field that
        changes sign between adjacent copies of the computational window.
    """
    def __init__(self, d_in, wavelength, frequency_offset=0.):
        super().__init__(d_in, wavelength)
        self.frequency_offset = frequency_offset

    def propagate(self, field, z):
        """Propagate the field over distance z using the Fresnel transfer function.

        Parameters
        ----------
        field : torch.Tensor
            Complex input field, shape (..., n, n).
        z : float or torch.Tensor
            Propagation distance [m]. May be positive, negative or zero;
            gradients are retained at zero distance.

        Returns
        -------
        torch.Tensor
            Propagated field with the same shape and sampling as the input.
        """
        n = field.shape[-1]
        freq = torch.fft.fftfreq(n, device=field.device, dtype=torch.float64) / self.d_in
        freq = freq + self.frequency_offset
        transfer_function = torch.exp(
            -1j * torch.pi * self.wavelength * z *
            (freq[:, None].square() + freq[None, :].square())
        ).to(field.dtype)
        axis = (torch.arange(n, device=field.device, dtype=torch.float64) - (n - 1) / 2) * self.d_in
        phase_factor = torch.exp(2j * torch.pi * self.frequency_offset *
                                 (axis[:, None] + axis[None, :])).to(field.dtype)
        input_field = torch.fft.ifftshift(field / phase_factor, dim=(-2, -1))
        output_field = torch.fft.ifft2(torch.fft.fft2(input_field) * transfer_function)
        return torch.fft.fftshift(output_field, dim=(-2, -1)) * phase_factor


class FresnelSingle(Fresnel):
    """Fresnel diffraction using a single Fourier transform.

    Sampling must resolve the input field including its quadratic phase,
    and the output quadratic phase. The propagation distance must be nonzero.

    Attributes
    ----------
    d_out : float or torch.Tensor
        Spatial sampling interval of the output field [m].
    n_out : int or None
        Number of output samples in each dimension. None uses the input size.
    transform : {'auto', 'fft', 'bluestein'}
        Numerical Fourier transform. Bluestein allows independent output
        sampling. FFT requires n_out = n_in and
        d_in*d_out = wavelength*abs(z)/n_in, with a fixed transform scale.
    """
    def __init__(self, d_in, wavelength, d_out, n_out=None, transform='auto'):
        super().__init__(d_in, wavelength)
        self.d_out = d_out
        self.n_out = n_out
        self.transform = transform

    def propagate(self, field, z):
        """Propagate the field using the single transform Fresnel integral.

        Parameters
        ----------
        field : torch.Tensor
            Complex input field, shape (..., n_in, n_in).
        z : float or torch.Tensor
            Nonzero propagation distance [m]. Use FresnelTransfer at z=0.

        Returns
        -------
        torch.Tensor
            Propagated field, shape (..., n_out, n_out), including the quadratic
            phase factors and Fresnel normalization d_in**2/(i*wavelength*z).
        """
        n_in = field.shape[-1]
        n_out = n_in if self.n_out is None else self.n_out
        transform = self.transform
        if float(z.detach() if torch.is_tensor(z) else z) == 0:
            raise ValueError('single-transform Fresnel diffraction requires nonzero z')
        n_fft = self.wavelength * z / (self.d_in * self.d_out)
        trainable = torch.is_tensor(n_fft) and n_fft.requires_grad
        n_fft_value = float(n_fft.detach() if torch.is_tensor(n_fft) else n_fft)
        natural_grid = n_out == n_in and math.isclose(abs(n_fft_value), n_in, rel_tol=1e-12)
        if transform == 'auto':
            transform = 'fft' if natural_grid and not trainable else 'bluestein'
        x = (torch.arange(n_in, device=field.device, dtype=torch.float64) - (n_in - 1) / 2) * self.d_in
        u = (torch.arange(n_out, device=field.device, dtype=torch.float64) - (n_out - 1) / 2) * self.d_out
        phase_in = torch.exp(1j * torch.pi / (self.wavelength * z) *
                             (x[:, None].square() + x[None, :].square())).to(field.dtype)
        input_field = field * phase_in
        if transform == 'bluestein':
            fourier_field = bluestein_fft(input_field, n_out, n_fft, inverse=True)
        elif transform == 'fft':
            if not natural_grid or trainable:
                raise ValueError('FFT requires a fixed natural output grid; use Bluestein otherwise')
            fourier_field = centered_fft(input_field, inverse=n_fft > 0)
        else:
            raise ValueError('transform must be auto, fft or bluestein')
        phase_out = torch.exp(1j * torch.pi / (self.wavelength * z) *
                              (u[:, None].square() + u[None, :].square())).to(field.dtype)
        return fourier_field * phase_out * (self.d_in**2 / (1j * self.wavelength * z))
