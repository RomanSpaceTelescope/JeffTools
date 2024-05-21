from __future__ import annotations

# from typing import List
from dataclasses import dataclass

import numpy as np

# from typing import Union
from astropy.table import Table

from .instrument import (
    GrismInstrument,
    ImageInstrument,
    InstrumentMode,
    PrismInstrument,
)

__all__ = ['Telescope', 'TelescopeOptics']


@dataclass
class Telescope:
    telescope: str
    instrument_name: str
    mode: str
    sca: int
    opt_elem: Table
    pm_diam: float
    """primary mirror diameter, meters"""
    sm_diam: float
    n_refl: int
    n_refr: int
    sys_flen: float
    sys_fratio: float
    cos_sca_tilt: np.ndarray
    filters: Table
    fdep_fnum_outer: Table
    """ filter-dependent values of outer f/# """
    fdep_fnum_inner: Table
    """ filter-dependent values of inner f/# """
    fdep_solid_angle: Table
    """ filter-dependent values of solid angle """
    det_bp_lo: float
    """ low edge of detector bandpass (microns)"""
    det_bp_hi: float
    """ high edge of detector bandpass (microns)"""
    r_jitter: float
    """ rms jitter in arcsec"""
    x_jitter: float
    """ rms x-axis jitter: in arcsec"""
    y_jitter: float
    """ rms y-axis jitter: in arcsec"""
    # psf_files: List[str]
    # """ paths to FITS files containing effective PSF to use"""
    r_other: np.ndarray
    """ rms of other PSF contributions: in arcsec"""
    pix_size: float
    """pixel size in microns"""
    pix_sc: float
    """mean pixel scale in arcsec"""
    pix_sc_x: float
    """x-axis pixel scale in arcsec"""
    pix_sc_y: float
    """y-axis pixel scale in arcsec"""
    full_well: float
    """full-well depth in electrons"""
    dark_cur: float
    """dark current at nominal temp, e/p/s"""
    frame_time: float
    """frame time in seconds"""
    mode_type: InstrumentMode
    """Telescope Mode Type"""
    # Default values.
    r_diffu: float = 2.0
    """charge diffusion length, in microns"""
    cds_readnoise: float = 15.0  # changed 2022 06 03 to match flight SCA data.
    """CDS readnoise in electrons/pix"""
    readnoise_floor: float = 5.0  # 2.5 Finger et al 2008 is best I've seen
    """minimum readnoise regardless of number of reads"""
    xtalk: float = 0.02
    """pixel to pixel cross-talk"""
    instrument: ImageInstrument | GrismInstrument | PrismInstrument | None = None


@dataclass
class TelescopeOptics:
    tel_elems: Table
    pm_diam: float
    sm_diam: float
    n_refl: int
    sys_focal_length: float
    focal_length_ratio: float
    cos_sca_tilt: np.ndarray
    pixel_scale: float
