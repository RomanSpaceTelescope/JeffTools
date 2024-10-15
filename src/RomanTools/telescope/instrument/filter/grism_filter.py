import numpy as np


def grism_filter_params(sca_index: int) -> dict:
    """
    Calculates spectral parameters for a slitless grism spectrometer based on the
    given sensor chip assembly (SCA) index. This includes defining the spectral
    calibration, blue and red edges of the bandpass, edge thickness, dispersion,
    and the wavelengths for sampling the point spread function (PSF).

    Parameters:
    - sca_index (int): Index of the sensor chip assembly element for which parameters
                       are to be calculated.

    Returns:
    - dict: A dictionary containing grism filter parameters such as nominal
            wavelengths ('low', 'high'), wavelength thickness ('wlow', 'whigh'),
            number of spectral samples ('n_spect'), wavelength step for PSF
            sampling ('w_sp_psf'), and dispersion ('grs_disp').
    """
    # Define Galaxy Redshift Survey slitless grism spectrometer parameter structure.
    # Define wavelength scale in advance, as we need to know the size of the
    # associated arrays.
    # define bandpass and band-edge thickness here.
    # values from "2022-08-11 Grism Flight spectral calibration - wrap up.pptx"

    # fmt: off
    w_blue_edge = np.array([0.99934, 0.99931, 0.99805, 0.99777, 0.99811, 0.99704,
                            0.99428, 0.99529, 0.99495, 0.99934, 0.99931, 0.99805,
                            0.99777, 0.99797, 0.99704, 0.99430, 0.99529, 0.99475
                            ])
    # fmt: on
    wmin_grs = w_blue_edge[sca_index]
    # fmt: off
    w_red_edge = np.array([1.9188, 1.9179, 1.9140, 1.9156, 1.9155, 1.9122, 1.9083,
                           1.9098, 1.9082, 1.9188, 1.9179, 1.9140, 1.9152, 1.9148,
                           1.9114, 1.9078, 1.9071, 1.9071
                           ])
    # fmt: on
    wmax_grs = w_red_edge[sca_index]

    # fmt: off
    s_blue_edge = np.array([0.0043, 0.0041, 0.0042, 0.0048, 0.0043, 0.0044, 0.0059,
                            0.0053, 0.0052, 0.0043, 0.0041, 0.0042, 0.0048, 0.0048,
                            0.0044, 0.0066, 0.0053, 0.0061
                            ])
    # fmt: on
    wthk_grs_lo = s_blue_edge[sca_index]

    # fmt: off
    s_red_edge = np.array([0.012, 0.012, 0.012, 0.012, 0.011, 0.012, 0.012, 0.012,
                           0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012,
                           0.013, 0.013
                           ])
    # fmt: on
    wthk_grs_hi = s_red_edge[sca_index]

    # do computations past nominal edge, by 2 edge thicknesses
    # wthk * wmin and wthk * wmax give the delta - lambda for 10 % -90 % transmission
    # go twice this far past wmin, wmax
    w_lo = wmin_grs * (1.0 - 3.0 * wthk_grs_lo)
    w_hi = wmax_grs * (1.0 + 3.0 * wthk_grs_hi)

    # rough dispersion is linear
    # varies somewhat under 1 % as function of wavelength
    # Can update this later with the non - linear terms, but the net effect is small
    # next array is dispersion in nm / pix
    # values from "2022-08-11 Grism Flight spectral calibration - wrap up.pptx"
    # fmt: off
    grs_disp_vs_sca = np.array([1.095, 1.057, 1.009, 1.096, 1.061, 1.014, 1.098,
                                1.067, 1.027, 1.095, 1.057, 1.009, 1.096, 1.060,
                                1.014, 1.098, 1.067, 1.026
                                ])
    # fmt: on
    grs_disp = grs_disp_vs_sca[sca_index] * 1e-3
    n_spect = np.ceil((w_hi - w_lo) / grs_disp).astype(int) + 1
    nf = 1

    # define wavelengths at which to sample the PSF
    # we will be using existing PSF files so set these values accordingly.
    w_sp_psf = np.arange(1, 2, 0.1)

    return {
        'name': 'GRS Grism',
        'nf': nf,
        'low': np.array([wmin_grs]),
        'high': np.array([wmax_grs]),
        'wlow': np.array([wthk_grs_lo * wmin_grs]),
        'whigh': np.array([wthk_grs_hi * wmax_grs]),
        'n_spect': np.array([n_spect]),
        'w_sp_psf': w_sp_psf,
        'grs_disp': np.array([grs_disp]),
    }
