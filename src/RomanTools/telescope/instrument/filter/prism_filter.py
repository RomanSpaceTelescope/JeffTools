from typing import Tuple

import numpy as np


def prism_filter_spectra(
    w_lo, w_hi, pixel_scale
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rfit_w0 = 0.7  # This parametrization matches dispersion curve from 2019 05 10
    rfit_w1 = 1.6
    rfit_dw0 = 0.3
    rfit_dw1 = 1.2
    rfit_r0 = 180.0
    rfit_r1 = 80.0
    resel = 2.0

    def resolv(w):
        res0 = rfit_r0 * np.exp(-(w - rfit_w0) / rfit_dw0)
        res1 = rfit_r1 * np.exp(+(w - rfit_w1) / rfit_dw1)
        return res0 + res1

    resel = 2.0
    # define spectral bins and resolution
    w_spect = [w_lo]
    d_lambda = [w_lo * 1.0 / (resel * resolv(w_spect[-1]))]
    r_theta = [w_lo * pixel_scale / d_lambda[0]]

    # for i = 1, n_spect - 1 do begin
    while w_spect[-1] <= w_hi:
        w_spect.append(w_spect[-1] + d_lambda[-1])
        d_lambda.append(w_spect[-1] * 1.0 / (resel * resolv(w_spect[-1])))
        r_theta.append(w_spect[-1] * pixel_scale / d_lambda[-1])

    return np.array(w_spect), np.array(r_theta), np.array(d_lambda)


def prism_filter_params(sca_index: int, pixel_scale: float) -> dict:
    """
    Calculates spectral parameters for a slitless prism spectrometer based on the
    given sensor chip assembly (SCA) index. This includes defining the spectral
    calibration, blue and red edges of the bandpass, edge thickness, dispersion,
    and the wavelengths for sampling the point spread function (PSF).

    Parameters:
    - sca_index (int): Index of the sensor chip assembly element for which parameters
                       are to be calculated.

    Returns:
    - dict: A dictionary containing prism filter parameters such as nominal
            wavelengths ('low', 'high'), wavelength thickness ('wlow', 'whigh'),
            number of spectral samples ('n_spect'), wavelength step for PSF
            sampling ('w_sp_psf'), and dispersion ('grs_disp').
    """
    # Define Supernova prism spectrometer parameter structure.
    # Define wavelength scale in advance, as we need to know the size of the
    # associated arrays.
    # define bandpass and band-edge thickness here.
    # values from "2022-08-18 Prism Flight spectral calibration - wrap up.pptx"

    # fmt: off
    w_blue_edge = np.array([0.76149, 0.75998, 0.75773, 0.76049, 0.75921, 0.75719,
                            0.75845, 0.75775, 0.75623, 0.76149, 0.75998, 0.75770,
                            0.76052, 0.75924, 0.75722, 0.75849, 0.75782, 0.75635
                            ])
    # fmt: on
    wmin_sn = w_blue_edge[sca_index]
    # fmt: off
    w_red_edge = np.array([1.8168, 1.8200, 1.8157, 1.8126, 1.8173, 1.8143, 1.8025,
                           1.8108, 1.8111, 1.8167, 1.8200, 1.8159, 1.8122, 1.8173,
                           1.8146, 1.8017, 1.8107, 1.8114
                            ])
    # fmt: on
    wmax_sn = w_red_edge[sca_index]

    # fmt: off
    s_blue_edge = np.array([0.0065, 0.0064, 0.0066, 0.0066, 0.0064, 0.0067, 0.0072,
                            0.0069, 0.0074, 0.0066, 0.0063, 0.0065, 0.0064, 0.0061,
                            0.0064, 0.0065, 0.0062, 0.0064
                            ])
    # fmt: on
    wthk_sn_lo = s_blue_edge[sca_index]

    # fmt: off
    s_red_edge = np.array([0.013, 0.008, 0.009, 0.014, 0.009, 0.009, 0.017, 0.010,
                           0.008, 0.013, 0.009, 0.009, 0.014, 0.009, 0.009, 0.017,
                           0.011, 0.009
                           ])
    # fmt: on
    wthk_sn_hi = s_red_edge[sca_index]

    # do computations past nominal edge, by 2 edge thicknesses
    # wthk * wmin and wthk * wmax give the delta - lambda for 10 % -90 % transmission
    # go twice this far past wmin, wmax
    w_lo = wmin_sn * (1.0 - 3.0 * wthk_sn_lo)
    w_hi = wmax_sn * (1.0 + 3.0 * wthk_sn_hi)

    nf = 1

    # define wavelengths at which to sample the PSF
    # we will be using existing PSF files so set these values accordingly.
    w_sp_psf = np.array([0.75, 0.88, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])

    w_spect, r_theta, d_lambda = prism_filter_spectra(w_lo, w_hi, pixel_scale)

    return {
        'name': 'PR-band',
        'nf': nf,
        'low': np.array([wmin_sn]),
        'high': np.array([wmax_sn]),
        'wlow': np.array([wthk_sn_lo * wmin_sn]),
        'whigh': np.array([wthk_sn_hi * wmax_sn]),
        'n_spect': len(w_spect),
        'w_sp_psf': w_sp_psf,
        'w_spect': w_spect,
        'r_theta': r_theta,
        'd_lambda': d_lambda,
    }
