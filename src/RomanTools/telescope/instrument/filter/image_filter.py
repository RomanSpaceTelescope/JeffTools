from importlib.resources import as_file, files

import numpy as np
from astropy.io.ascii import read
from astropy.table import vstack


def image_filter_params(
    idx: int, img_dw: float = 0.005, filt_tmax: float = 0.95
) -> dict:
    """
    Retrieves and processes bandpass filter parameters for the Roman Telescope based on
    the specified sensor chip assembly (SCA) index. It reads from predefined data files,
    stacks them, and calculates wavelength ranges for each filter. The output includes
    nominal wavelengths, range adjustments, number of spectral samples, and a constant
    transmission value for all filters.

    Parameters:
    - idx (int): Index of the sensor chip assembly.
    - img_dw (float): Delta wavelength for computing the number of spectral samples.
    - filt_tmax (float): Maximum transmission value for filters.

    Returns:
    - dict: Dictionary containing filter parameters and wavelength information.
    """

    # Bandpass filters
    with as_file(files('RomanTools.data') / 'skinny_us220240305_bandpasses.txt') as f:
        bptable = read(f, header_start=1, data_start=2)

    with as_file(files('RomanTools.data') / 'F184_us220240305_bandpasses.txt') as f:
        bptable = vstack([bptable, read(f, header_start=1, data_start=2)])

    bptable = bptable[bptable['SCA'] == idx]

    filt_name = bptable['Filt'].data
    nf = len(bptable)
    assert nf == 8

    # Create arrays of wavelengths for sampling spectra within filter bandpass
    filter_lo = 1e-3 * bptable['W(blue)']
    filter_dlo = 1e-3 * bptable['dW(blue)']
    filter_hi = 1e-3 * bptable['W(red)']
    filter_dhi = 1e-3 * bptable['dW(red)']
    wlo = filter_lo - 2.0 * filter_dlo
    whi = filter_hi + 2.0 * filter_dhi
    n_wave = np.ceil((whi - wlo) / img_dw).astype(int) + 1

    # Generate wavelength samples for each filter
    wavelengths = np.zeros((nf, np.max(n_wave)))
    for i in range(nf):
        wavelengths[i, : n_wave[i]] = np.linspace(wlo[i], whi[i], n_wave[i])

    # filt_tr = np.full(8, filt_tmax)

    return {
        'name': filt_name,
        'nf': nf,
        'low': filter_lo,
        # 'dlow': filter_dlo,
        'high': filter_hi,
        # 'dhigh': filter_dhi,
        'wlow': wlo,
        'whigh': whi,
        'n_wave': n_wave,
        'wavelengths': wavelengths,
        # 'tr': filt_tr,
        'transmission': np.full(nf, filt_tmax),
    }
