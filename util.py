"""Image cutouts and flux measurements for the PyFLIP runners."""


def crop_image(image, size, center=None):
    """Extract a square window from the last two image dimensions.

    Parameters
    ----------
    image : numpy.ndarray or torch.Tensor
        Image or image stack.
    size : int
        Window width [pixels]. Odd widths round down to an even width.
    center : tuple of int, optional
        Window center as (row, column) [pixels]. Defaults to the image center.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Image view with leading dimensions preserved. Bounds use ordinary
        slicing, without padding or resampling.
    """
    if center is None:
        center = (image.shape[-2] // 2, image.shape[-1] // 2)
    row, column = center
    half_size = size // 2
    return image[..., row-half_size:row+half_size, column-half_size:column+half_size]


def masked_flux(image, mask, size):
    """Sum masked intensity within a central square window.

    Parameters
    ----------
    image, mask : numpy.ndarray or torch.Tensor
        Images and mask on the same spatial grid. Leading dimensions may
        broadcast; the sum includes all images in a stack.
    size : int
        Window width [pixels], using the convention of ``crop_image``.

    Returns
    -------
    scalar or torch.Tensor
        Total masked intensity, preserving Torch gradients when applicable.
    """
    return (crop_image(mask, size) * crop_image(image, size)).sum()
