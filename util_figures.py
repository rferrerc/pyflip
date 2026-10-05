"""Figures, progress videos and diagnostic exports for the PyFLIP runners."""
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def save_image(image, filename):
    """Save a 2D image using the runner's colormap and lower-left origin."""
    plt.imsave(filename, image, cmap='viridis', origin='lower')


def plot_injected_planet(image, filename):
    """Save the simulated companion image with its summed and peak intensity."""
    plt.figure(figsize=[5, 5])
    plt.imshow(np.squeeze(image), origin='lower')
    plt.title(f'Sum, peak = {np.nansum(image):.2f},{np.nanmax(image):.2f} (sim planet)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_planet_diagnostics(image, masked_image, annulus, signal_blob, peak_blob,
                            companion, snr, filename):
    """Save the residual, noise annulus and companion signal in one figure.

    Parameters
    ----------
    image, masked_image, annulus, signal_blob : numpy.ndarray
        Residual and diagnostic images from the companion measurement.
    peak_blob : float
        Measured peak companion intensity.
    companion : dict
        Injected companion properties: pos_x_px, pos_y_px and flux.
    snr : float
        Measured signal-to-noise ratio.
    filename : str or pathlib.Path
        Output figure filename.
    """
    plt.figure(figsize=[15, 10])
    plt.subplot(231)
    plt.imshow(image, origin='lower')
    plt.title(f'Sum, peak = {np.nansum(image):.2f},{np.nanmax(image):.2f} (orig)')
    plt.subplot(232)
    plt.imshow(masked_image, origin='lower', vmin=-10)
    plt.subplot(234)
    plt.imshow(annulus, origin='lower')
    plt.title(f'Stddev = {np.nanstd(annulus):.2f} (annulus synthetic)')
    plt.subplot(235)
    plt.imshow(signal_blob, origin='lower')
    plt.title(f'Sum, peak = {np.nansum(signal_blob):.2f},{peak_blob:.2f},\n pos x, pos y, flux input ={companion["pos_x_px"]:.2f},{companion["pos_y_px"]:.2f},{companion["flux"]:.2f}')
    whole_annulus = np.where(~np.isnan(signal_blob), signal_blob, annulus)
    plt.subplot(236)
    plt.imshow(whole_annulus, origin='lower')
    plt.title(f'Whole annulus. \n Peak, mean, stddev = {np.nanmax(whole_annulus):.2f}, {np.nanmean(whole_annulus):.2f}, {np.nanstd(whole_annulus):.2f}')
    plt.suptitle(f'Curr SNR: {snr:.2f}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_progress_video(images, filename):
    """Save residual images as a video with each frame scaled to its own range.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        2D residual images in iteration order.
    filename : str or pathlib.Path
        Output MP4 filename.
    """
    frames = np.array(images)
    frames = np.array([(im - im.min()) / (im.max() - im.min()) for im in frames])
    frames = np.uint8(cm.viridis(frames) * 255)
    frames = np.flip(frames, 1)
    imageio.mimsave(filename, frames, 'FFMPEG', macro_block_size=None,
                    ffmpeg_params=['-s', '256x256', '-v', '0'], fps=30)


def save_planet_metrics(output_dir, peaks, signal_loss, target_snr, injected_snr, iterations):
    """Save companion measurement histories using the existing output filenames.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Existing output directory.
    peaks, signal_loss, target_snr, injected_snr, iterations : array-like
        Peak intensities, recovered flux fractions, target and injected-companion
        signal-to-noise ratios, and the corresponding iteration numbers.
    """
    output_dir = Path(output_dir)
    np.save(output_dir / 'sci_injected_peaks_postsub.npy', np.asarray(peaks))
    np.save(output_dir / 'sci_injected_signal_loss.npy', np.asarray(signal_loss))
    np.save(output_dir / 'realtarget_snr.npy', np.asarray(target_snr))
    np.save(output_dir / 'sci_injected_snr.npy', np.asarray(injected_snr))
    np.save(output_dir / 'sci_iters.npy', np.asarray(iterations))


def save_fit_results(output_dir, reference_images, target_images, oversample, num_wl):
    """Save reference/science progress videos and the final residual arrays.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Existing output directory.
    reference_images, target_images : sequence of numpy.ndarray
        Reference and science residual histories.
    oversample, num_wl : int
        Oversampling factor and wavelength count used in the output filenames.
    """
    output_dir = Path(output_dir)
    save_progress_video(reference_images, output_dir / 'progress_REFERENCE.mp4')
    save_progress_video(target_images, output_dir / 'progress_TARGET.mp4')
    target_arrays = np.squeeze(np.array(target_images))
    reference_arrays = np.squeeze(np.array(reference_images))
    np.save(output_dir / f'last_iteration_oversample_{oversample}_wl_sampling_{num_wl}.npy', target_arrays[-1])
    np.save(output_dir / f'last_iteration_REFERENCE_oversample_{oversample}_wl_sampling_{num_wl}.npy', reference_arrays[-1])
