import numpy as np
from astropy import units as u

__all__ = ['r_other']


def r_other(wfe_wref, wfe_nm, pm_diam):
    """
    Calculate the RMS of other point spread function (PSF) contributions.

    Computes the root mean square (RMS) value representing other PSF
    contributions in the system, which are not accounted for by main PSF
    effects, within a specified filter range and primary mirror diameter.

    Returns:
        float: The RMS of other PSF contributions measured in arcseconds.
    """
    wfe_waves = wfe_nm / wfe_wref

    # set r_other based on wavefront error
    wfe_scalefac = 2.0 * np.exp(np.sqrt(0.5) * (2.0 * np.pi * wfe_waves))
    # exponent e-9 in next line comes from fact that wfe_wref is in nm and pm_diam is in meters
    fwhm_diffr_nominal = (1.03e-9 * wfe_wref / pm_diam * u.rad).to(u.arcsec).value
    wfe_rms = fwhm_diffr_nominal * np.sqrt(wfe_scalefac**2 - 1.0) / 2.354
    # apply fudge factor so that number of effective pixels under the PSF matches what
    # comes from a PSF calculated by WebbPSF. This accounts for the fact that this
    # analytic expression doesn't move the light as far out from the core as reality.
    r_other_scalefac = 1.5
    r_other = wfe_rms * r_other_scalefac
    return r_other
