from dataclasses import dataclass

import numpy as np

__all__ = ['ImageInstrument', 'GrismInstrument', 'PrismInstrument']


@dataclass
class ImageInstrument:
    """Imager Instrument class."""

    w_spect: np.ndarray  # wavelengths at which to sample spectrum w/in bandpass


@dataclass
class GrismInstrument:
    """Grism Instrument class."""

    w_spect: float  # wavelength for each spectral pixel
    r_theta: float  # resolving power
    d_lambda: float  # width of each pixel in wavelength space
    blaze_angle: float = 2.32  # blaze angle is 1.53 degrees


@dataclass
class PrismInstrument:
    """Prism Instrument class."""

    w_spect: np.ndarray  # wavelength for each spectral pixel
    r_theta: np.ndarray  # resolving power
    d_lambda: np.ndarray  # width of each pixel in wavelength space
