"""Small CPU fixtures for the refactored optical classes, not the legacy runner."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

# The refactor uses sibling imports and is not an installed package. Select it
# explicitly rather than accidentally testing the root-level legacy modules.
REFACTOR = Path(__file__).resolve().parents[1] / "refactoring"
sys.path.insert(0, str(REFACTOR))

import dl_utils
import model_classes
import optical_elements
import wavefronts

for module in (dl_utils, model_classes, optical_elements, wavefronts):
    assert Path(module.__file__).resolve().parent == REFACTOR


@pytest.fixture
def optical_case(monkeypatch):
    # DiffDetector(device='cpu') does not forward its device to FlatFieldingModule.
    # Override only this constructor default, retaining the real implementation.
    # This fixture does NOT establish that production CPU construction works.
    monkeypatch.setattr(
        optical_elements, "FlatFieldingModule",
        lambda h, w: model_classes.FlatFieldingModule(h, w, device="cpu"),
    )
    n, n_det, oversample = 8, 6, 2
    diameter = 6.603464
    wavelengths = np.array([3.9e-6, 4.4e-6, 4.9e-6])
    weights = np.array([0.17, 0.29, 0.54])
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n, dtype=torch.float64),
        torch.linspace(-1, 1, n, dtype=torch.float64), indexing="ij",
    )
    aperture = (((x + 0.12)**2 + (1.15*y)**2 < 0.95) *
                (0.8 + 0.08*x - 0.03*y))[None]
    lyot = (((x - 0.08)**2 + (1.25*y)**2 < 0.72) *
            (0.9 - 0.05*x + 0.02*y))[None]
    entrance_opd = 35e-9 * (x**2 - 0.4*y + x*y)
    internal_opd = torch.stack([
        (i + 1) * 12e-9 * (y**2 + 0.2*x + 0.3*x*y)
        for i in range(len(wavelengths))
    ])[None]
    f = (torch.arange(n*oversample, dtype=torch.float64) - n) / n
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    fpm = torch.stack([
        1 - (0.8 + 0.03*i) * torch.exp(-((fx - 0.06)**2 + (fy + 0.04)**2) / (0.1 + 0.02*i))
        for i in range(len(wavelengths))
    ])[None]
    wf = wavefronts.BroadbandWavefront(
        n, diameter, np.stack([wavelengths, weights]),
        torch.tensor([2.3], dtype=torch.float64),
        angles=torch.tensor([2.1e-8, -1.3e-8], dtype=torch.float64), DEVICE="cpu",
    ).double()
    primary = optical_elements.DiffPupilOptic(
        opd=entrance_opd, amplitude=aperture,
        wfe_offsets=model_classes.OPDOffsetModule(n, n, device="cpu").double(),
    )
    stop = optical_elements.DiffPupilOptic(
        opd=internal_opd, amplitude=lyot,
        wfe_offsets=model_classes.OPDOffsetModule(n, n, device="cpu").double(),
    )
    args = (n, wf.wavelengths, diameter/n, n_det, dl_utils.arcsec2rad(0.062424185),
            None, [0.0, 0.0], True, False)
    detector = optical_elements.DiffDetector(
        args, oversample=oversample, num_det_px=n_det, device="cpu",
    ).double()
    image_optic = optical_elements.DiffImageOptic(amplitude=fpm)
    system = optical_elements.DiffOpticalSystem(
        optical_element_list=[primary, image_optic, stop, detector], oversample=oversample,
    )
    return SimpleNamespace(
        n=n, n_det=n_det, oversample=oversample, wf=wf, primary=primary,
        image_optic=image_optic, stop=stop, detector=detector, system=system, args=args,
    )
