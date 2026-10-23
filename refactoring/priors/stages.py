import numpy as np
import torch

# optimization_handlers.default_stages with one special quirk: an additional flag fit_ote_in_science
# if False, this is the default_stages behaviour; if True, the OTE OPD is also optimized in science stage

def prior_stages(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000], fit_ote_in_science=False):
    # Zero-out gradients of parameters we are not optimizing per stage
    if stage_number in stage_cutoffs[:2]:
        # default_stages zeroes nothing at an iteration equal to a cutoff, so the star position (not fitted between
        # stage_cutoffs[0] and stage_cutoffs[2]) would take one unzeroed gradient here; with a poor fit it dominates
        # Adam's decayed second moment and kicks the position by tens of learning rates (seen with a frozen Lyot plane)
        wavefront.angles_offset.grad *= 0.
    if stage_number < stage_cutoffs[0]:
        # Only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.

    elif stage_number > stage_cutoffs[0] and stage_number < stage_cutoffs[1]:
        # Stop fitting position and flux; only fit OPDs
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.

    elif stage_number > stage_cutoffs[1] and stage_number < stage_cutoffs[2]:
        # Fit OPDs and detector flats
        wavefront.angles_offset.grad*=0
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 1.

    elif stage_number > stage_cutoffs[2] and stage_number < stage_cutoffs[3]:
        # Now we should be on the science star, so again only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.

    elif stage_number > stage_cutoffs[3]:
        # Fitting only one OPD in science star (two with fit_ote_in_science), and stop fitting position
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 1. if fit_ote_in_science else 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.


# Stages of testing_sci.py / testing_sci_priors.py (no reference frame): star position only before stage_cutoffs[0], then the
# two OPD grids (or, with the priors, their latents: the grids are frozen by DPSAdamOPD's hook); the detector flats are never
# fitted (a per-pixel flat fitted on the planet frame could absorb the planet).

def sci_stages(instrument, wavefront, stage_number, stage_cutoffs = [600, 5500]):
    instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.
    if stage_number < stage_cutoffs[0]:
        # Only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
    else:
        # Stop fitting position and flux; only fit OPDs
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
