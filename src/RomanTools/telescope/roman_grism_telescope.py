from importlib.resources import as_file, files

import numpy as np
from astropy.io.ascii import read
from astropy.table import vstack

from . import RomanInputParams
from .instrument import (
    GrismInstrument,
    InstrumentMode,
    downstream_elements,
    upstream_elements,
)
from .instrument.filter import grism_filter_params
from .instrument.filter.filter_dep import fdep_trio
from .psf import r_other
from .pupil_obscuration_geometry import pupil_obscuration_geom
from .roman_optics import roman_optics
from .sca_dependent_data import (
    sca_dep_full_well,
    sca_dep_xavg_pixel_scale,
    sca_dep_yavg_pixel_scale,
)
from .tel_elem_table import elements_table
from .telescope import Table, Telescope


def roman_grism_telescope(
    sca: int,
    params: RomanInputParams,
) -> Telescope:
    """
    Configures the Roman Wide-Field Instrument for grism mode, including the setup of
    telescope optical elements based on the specific SCA index.

    Args:
    - sca (int): Sensor chip assembly index.
    - params (RomanInputParams): Configuration parameters for the instrument.

    Returns:
    - Telescope: A configured Telescope object for grism mode with detailed setup
                 of filters, pupil geometry, and focal properties.

    Raises:
    - KeyError: If the SCA index is out of the accepted range [0,17].

    This function integrates multiple components such as optical elements, filters,
    and computational geometry for effective telescope simulation in grism mode.
    """
    if sca < 0 or sca >= 18:
        raise KeyError(f'sca index outside range [0,17]: {sca}')

    optics = roman_optics(sca, params)
    elems_us = upstream_elements(params, optics['tel_elems']['Obscuration Azim Frac'])
    elems_ss = elements_table(
        names=['Filter', 'P1', 'P2'],
        types=['filter', 'grism', 'grism'],
        materials=['CaF2', 'CaF2', 'CaF2'],
        emissivities=np.full(3, params.em_glass),
        temperatures=np.full(3, params.filter_temp),
        fnum_outer=np.full(3, params.aperture_fni),
        fnum_inner=np.full(3, params.sfnid),
        tput_factor=np.ones(3),
        obsc_azimfrac=np.ones(3),
        th_emis_tput=np.ones(3),
        fdep=np.full(3, False, dtype=bool),
        zone=3,
    )
    elems_ds = downstream_elements(params)

    opt_elem = vstack([optics['tel_elems'], elems_us, elems_ss, elems_ds])
    sys_focal_length = optics['sys_focal_length']

    filter_pars = grism_filter_params(sca)

    # get pupil obstruction corresponding to sca = sca_index
    with as_file(
        files('RomanTools.data.throughput') / 'Grism_mask_throughput.txt'
    ) as f:
        data = read(f).columns[1].data
        obsc_filt_pupil = np.array([data[sca]])

    ############
    # Project mask dimensions back to entrance pupil, convert to meters
    # Only one filter in grism mode  - has full mask
    #                        GRS
    expupil_rim_od = params.aperture_id
    #
    # baseline as of 2020 04 22 & OMD Rev E.
    expupil_rim_id = 2e-3 * np.array([44.5])
    expupil_center_od = 2e-3 * np.array([13.8])
    expupil_leg_width = 1e-3 * np.array([2.4])

    pog = pupil_obscuration_geom(
        params,
        expupil_rim_id,
        expupil_rim_od,
        expupil_leg_width,
    )

    filters = Table(
        [
            [
                filter_pars['name'],
            ],
            filter_pars['low'],
            filter_pars['wlow'],
            filter_pars['high'],
            filter_pars['whigh'],
            np.full(filter_pars['nf'], expupil_rim_od),
            expupil_rim_id,
            expupil_center_od,
            expupil_leg_width,
            obsc_filt_pupil,
            pog['obsc_filt_pupil_geom'],
        ],
        names=(
            'Filter Name',
            'Low',
            'Low_width',
            'High',
            'High_width',
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

    w_spect = filter_pars['wlow']
    r_theta = filter_pars['wlow'] * optics['pixel_scale'] / filter_pars['grs_disp']
    d_lambda = filter_pars['grs_disp']

    # present nominal grism mode WFE budget is 146nm.
    wfe_wref = 1000 * w_spect
    wfe_nm = 146.0
    rother = r_other(wfe_wref, wfe_nm, params.pm_d)

    return Telescope(
        telescope='Roman OTA',
        instrument_name='Wide-Field Instrument',
        mode='GRS grism',
        sca=sca,
        opt_elem=opt_elem,
        pm_diam=params.pm_d,
        sm_diam=params.sm_d,
        n_refl=params.n_refl,
        n_refr=4,  # filter is on one surface of grism
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
        r_jitter=0.014,
        x_jitter=0.012,
        y_jitter=0.012,
        r_other=rother,
        pix_size=params.pixel_dimension,
        pix_sc=optics['pixel_scale'],
        pix_sc_x=sca_dep_xavg_pixel_scale[sca],
        pix_sc_y=sca_dep_yavg_pixel_scale[sca],
        full_well=sca_dep_full_well[sca],
        mode_type=InstrumentMode.PrismSpec,
        dark_cur=params.dark_current,
        frame_time=4.42,
        instrument=GrismInstrument(w_spect=w_spect, r_theta=r_theta, d_lambda=d_lambda),
    )
