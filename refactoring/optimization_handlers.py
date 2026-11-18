import numpy as np 
import torch 

# basic handler assuming the simplest scene, default instrument parameters, and default parameter names
def default_stages(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    # Zero-out gradients of parameters we are not optimizing per stage
    # Technically we don't need to multiply by 1, but keep for now for clarity and see if it slows down the code too much
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
        # Fitting only one OPD in science star, and stop fitting position
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad*= 0.    

# basic handler assuming the simplest scene, default instrument parameters, and default parameter names
def default_stages_NoneGrads(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    # Zero-out gradients of parameters we are not optimizing per stage
    # Technically we don't need to multiply by 1, but keep for now for clarity and see if it slows down the code too much
    if stage_number < stage_cutoffs[0]:
        # Only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad = None 
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad = None
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad = None
    
    elif stage_number > stage_cutoffs[0] and stage_number < stage_cutoffs[1]:
        # Stop fitting position and flux; only fit OPDs
        wavefront.angles_offset.grad = None
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad = None

    elif stage_number > stage_cutoffs[1] and stage_number < stage_cutoffs[2]:
        # Fit OPDs and detector flats
        wavefront.angles_offset.grad = None
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
        # Fitting only one OPD in science star, and stop fitting position
        wavefront.angles_offset.grad = None
        instrument.optical_system.JWSTPrimary.wfe_offsets.grid.grad = None 
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
        instrument.optical_system.NIRCamdetector.detector_flat_field.grid.grad = None 

def pos_and_lyot_stages(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    if stage_number < stage_cutoffs[0]:
        # Only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 0.

    if stage_number > stage_cutoffs[0] and stage_number < stage_cutoffs[1]:
        # Only fit star position
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 1.

def cycling_pos_and_lyot_stages(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    
    batch = int(np.ceil(stage_number/100))

    if batch%2==1:
        # Only fit star position
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 0.

    else:
        # Only fit star position
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 1.


def pos_and_lyot_all(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    
    batch = int(np.ceil(stage_number/100))

    # Only fit star position
    wavefront.angles_offset.grad*=1.
    instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 1.
    instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 1.


def first_onlypos_then_onlyrest(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    
    # batch = int(np.ceil(stage_number/100))

    # Only fit star position
    if stage_number < 0:
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
    else:#:
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.
    # else:
    #     wavefront.angles_offset.grad*=0.
    #     instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 0.
    #     instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 0.
    #     instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.


def fit_separately(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    
    # batch = int(np.ceil(stage_number/100))

    # Only fit star position
    if stage_number < 300:
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
    elif stage_number < 2200:
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
    else:
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.

def looping_pos_opd_lyot(instrument, wavefront, stage_number, stage_cutoffs = [600, 3500, 5500, 6100, 11000]):
    
    batch = int(np.ceil(stage_number/100))

    # Only fit star position
    if batch%3==0:
        wavefront.angles_offset.grad*=1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
    elif batch%3==1:
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 1.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 0.
    else:
        wavefront.angles_offset.grad*=0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.x_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.pupil_shifts.y_shift.grad*= 0.
        instrument.optical_system.NIRCamLyotstop.wfe_offsets.grid.grad*= 1.