from importlib.resources import as_file, files

import numpy as np
from astropy.table import vstack
from astropy.io.ascii import read

from . import RomanInputParams
from .roman_optics import roman_optics
from ..telescope import Telescope, Table
from ..tel_elem_table import elements_table
from ..instrument import (
    InstrumentMode,
    PrismInstrument,
    upstream_elements,
    downstream_elements,
)
from ..instrument.filter import prism_filter_params, fdep_trio
from .pupil_obscuration_geometry import pupil_obscuration_geom
from ..psf import r_other
from .sca_dependent_data import (
    sca_dep_xavg_pixel_scale,
    sca_dep_yavg_pixel_scale,
    sca_dep_full_well,
)


def roman_prism_telescope(
    sca: int,
    params: RomanInputParams,
) -> Telescope:
    """
    Configures the Roman Wide-Field Instrument for prism mode, including the setup of
    telescope optical elements based on the specific SCA index.

    Args:
    - sca (int): Sensor chip assembly index.
    - params (RomanInputParams): Configuration parameters for the instrument.

    Returns:
    - Telescope: A configured Telescope object for prism mode with detailed setup
                 of filters, pupil geometry, and focal properties.

    Raises:
    - KeyError: If the SCA index is out of the accepted range [0,17].

    This function integrates multiple components such as optical elements, filters,
    and computational geometry for effective telescope simulation in prism mode.
    """
    if sca < 0 or sca >= 18:
        raise KeyError(f'sca index outside range [0,17]: {sca}')

    optics = roman_optics(sca, params)
    elems_us = upstream_elements(params, optics['tel_elems']['Obscuration_azim_frac'])
    elems_ss = elements_table(
        names=['Filter', 'P1'],
        types=['filter', 'prism'],
        materials=['CaF2', 'CaF2'],
        emissivities=np.array([params.em_glass, params.em_glass]),
        temperatures=np.array([params.filter_temp, params.filter_temp]),
        fnum_outer=np.array([params.aperture_fni, params.aperture_fni]),
        fnum_inner=np.array([params.sfnid, params.sfnid]),
        tput_factor=np.ones(2),
        obsc_azimfrac=np.ones(2),
        th_emis_tput=np.ones(2),
        fdep=np.full(2, False, dtype=bool),
        zone=3,
    )
    elems_ds = downstream_elements(params)

    opt_elem = vstack([optics['tel_elems'], elems_us, elems_ss, elems_ds])
    sys_focal_length = optics['sys_focal_length']

    filter_pars = prism_filter_params(sca, optics['pixel_scale'])

    w_spect = filter_pars['w_spect']
    r_theta = filter_pars['r_theta']
    d_lambda = filter_pars['d_lambda']

    # present nominal prism mode WFE budget is 146nm.
    wfe_wref = 1000 * w_spect
    wfe_nm = np.full_like(wfe_wref, 100.0)
    rother = r_other(wfe_wref, wfe_nm, params.pm_d)

    # get pupil obstruction corresponding to sca = sca_index
    with as_file(
        files('RomanTools.data.throughput') / 'Prism_mask_throughput.txt'
    ) as f:
        data = read(f, format='no_header', data_start=0)
        obsc_filt_pupil = 1.0 - 1e-2 * np.array([data[sca][1]])

    ############
    # Project mask dimensions back to entrance pupil, convert to meters
    # Only one filter in prism mode  - has full mask
    #                        GRS
    expupil_rim_od = params.aperture_id
    expupil_rim_id = 2e-3 * np.array([47.5])
    expupil_center_od = 2e-3 * np.array([0.0])
    expupil_leg_width = 1e-3 * np.array([0.0])

    pog = pupil_obscuration_geom(
        params,
        expupil_rim_id,
        expupil_rim_od,
        expupil_leg_width,
    )

    filters = Table(
        [
            filter_pars['name'],
            filter_pars['low'],
            filter_pars['wlow'],
            filter_pars['high'],
            filter_pars['whigh'],
            params.filt_tmax * 0.99,
            np.full(filter_pars['nf'], expupil_rim_od),
            expupil_rim_id,
            expupil_center_od,
            expupil_leg_width,
            obsc_filt_pupil,
            pog['obsc_filt_pupil_geom'],
        ],
        names=(
            'Filter',
            'Low',
            'Low_width',
            'High',
            'High_width',
            'Transmission',
            'Expupil Rim Outer Diam',
            'Expupil Rim Inner Diam',
            'Expupil Center Outer Diam',
            'Expupil Leg Width',
            'Pupil Obscuration',
            'Relative Pupil Obscuration',
        ),
    )

    fdep_fnum_outer_tab, fdep_fnum_inner_tab, fdep_solid_angle_tab = fdep_trio(
        params,
        sys_focal_length,
        opt_elem,
        filter_pars['nf'],
        filter_pars['name'],
        expupil_rim_od,
        expupil_rim_id,
        expupil_center_od,
        expupil_leg_width,
        pog['mask_leg_azim_frac'],
        pog['mask_leg_width'],
    )

    return Telescope(
        telescope='Roman OTA',
        instrument_name='2 element slitless prism',
        mode='R=85 prism',
        sca=sca,
        opt_elem=opt_elem,
        pm_diam=params.pm_d,
        sm_diam=params.sm_d,
        n_refl=params.n_refl,
        n_refr=2,
        sys_flen=sys_focal_length,
        sys_fratio=sys_focal_length / params.pm_d,
        cos_sca_tilt=optics['cos_sca_tilt'],
        filters=filters,
        fdep_fnum_outer=fdep_fnum_outer_tab,
        fdep_fnum_inner=fdep_fnum_inner_tab,
        fdep_solid_angle=fdep_solid_angle_tab,
        det_bp_lo=0.3,
        det_bp_hi=2.6,
        # guiding off images of stars through broad-band filters
        # changed 2022 06 03 - actual spec is 0.008; 0.012 allows for roll contribution also
        # meaningless if PSF computed by WebbPSF
        r_jitter=0.043,
        x_jitter=0.04,  # intermediate case found by Gary Welter
        y_jitter=0.014,  # includes high-frequency component, not just LOS drift
        r_other=rother,
        pix_size=params.pixel_dimension,
        pix_sc=optics['pixel_scale'],
        pix_sc_x=sca_dep_xavg_pixel_scale[sca],
        pix_sc_y=sca_dep_yavg_pixel_scale[sca],
        full_well=sca_dep_full_well[sca],
        mode_type=InstrumentMode.PrismSpec,
        dark_cur=params.dark_current,
        frame_time=4.42,
        instrument=PrismInstrument(w_spect=w_spect, r_theta=r_theta, d_lambda=d_lambda),
    )
