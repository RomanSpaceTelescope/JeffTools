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
    ImageInstrument,
    upstream_elements,
    downstream_elements,
)
from ..instrument.filter import image_filter_params
from ..instrument.filter.filter_dep import fdep_trio
from .pupil_obscuration_geometry import pupil_obscuration_geom
from ..psf import r_other
from .sca_dependent_data import (
    sca_dep_xavg_pixel_scale,
    sca_dep_yavg_pixel_scale,
    sca_dep_full_well,
)


def roman_image_telescope(
    sca: int,
    params: RomanInputParams,
) -> Telescope:
    """
    Configures and returns the Roman telescope imaging setup for a given SCA.

    Args:
    - sca (int): Sensor chip assembly index, must be between 0 and 17.
    - params (RomanInputParams): Configuration parameters for the telescope.

    Returns:
    - Telescope: Configured telescope object for the Roman Wide-Field Instrument.

    Raises:
    - KeyError: If the provided SCA index is outside the valid range.

    This function integrates various subsystems such as optics, pupil obscuration, and
    filter parameters to fully define the imaging characteristics of the telescope
    for a specified SCA. It leverages extensive calculations and data table operations
    to set up the optical path, calculate the f-number, solid angle, and prepare
    filter-specific data tables.
    """
    if sca < 0 or sca >= 18:
        raise KeyError(f'sca index outside range [0,17]: {sca}')

    optics = roman_optics(sca, params)
    elems_us = upstream_elements(params, optics['tel_elems']['Obscuration_azim_frac'])
    elems_ss = elements_table(
        names=['Filter'],
        types=['filter'],
        materials=['fused silica'],
        emissivities=np.array([params.em_glass]),
        temperatures=np.array([params.filter_temp]),
        fnum_outer=np.array([params.aperture_fni]),
        fnum_inner=np.array([params.sfnid]),
        tput_factor=np.ones(1),
        obsc_azimfrac=np.ones(1),
        th_emis_tput=np.ones(1),
        fdep=np.full(1, False, dtype=bool),
        zone=3,
    )
    elems_ds = downstream_elements(params)

    opt_elem = vstack([optics['tel_elems'], elems_us, elems_ss, elems_ds])
    sys_focal_length = optics['sys_focal_length']

    # # Get modeled bandpass edge properties, models as of 2024 03 12
    filter_pars = image_filter_params(sca, params.img_dw, params.filt_tmax)

    # get pupil obstruction corresponding to sca = sca_index
    # short wave filters - convert from percent throughput to fractional obscuration
    with as_file(
        files('RomanTools.data.throughput') / 'Skinny_mask_throughput.txt'
    ) as f:
        data = read(f, format='no_header', data_start=0)
        ob_sw = 1.0 - 0.01 * data[sca]
    # long-wave filters
    with as_file(files('RomanTools.data.throughput') / 'F184_mask_throughput.txt') as f:
        data = read(f, format='no_header', data_start=0)
        ob_lw = 1.0 - 0.01 * data[sca]

    obsc_filt_pupil = np.array([ob_sw, ob_sw, ob_sw, ob_sw, ob_sw, ob_lw, ob_sw, ob_lw])

    expupil_rim_od = params.aperture_od
    #                                 F062  F087  F106  F129  F158  F184  W146  F213
    expupil_rim_id = 2e-3 * np.array([47.5, 47.5, 47.5, 47.5, 47.5, 44.3, 47.5, 44.3])
    expupil_center_od = 2e-3 * np.array(
        [12.0, 12.0, 12.0, 12.0, 12.0, 17.1, 12.0, 17.1]
    )
    expupil_leg_width = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 3.0]) / 1000.0

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
            filter_pars['transmission'],
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

    # # if user specfied a fov, then popuilate the PSF_FILES entry in the structure
    # # Define the constant parts of the file name pattern
    # psfdir = files("RomanTools.data.PSF")
    # prefix = 'WFI_PSF_'
    # # K0V spectral type, 8mas jitter, centered on pixels.
    # suffix = '_x2048_y2048_j008mas_os8_K0V_o.fits'
    # sca_str = f'_SCA{sca+1:02d}'  # SCA ID to two digits
    # psf_files = [f'{psfdir / prefix / filt_name[i] /
    #                 sca_str / suffix}' for i in range(nf)] if sca else []
    #                 # sca_str / suffix}' for i in range(nf)] if scap.fov else []

    # set nominal wavefront error at 14th of a wave at 1.2 microns.
    # This gives 85.7nm, almost exactly what is modeled in the zernike decomposition
    # in the spreadsheet from Bert Pasquale, Cathy Marx, and Guangjou
    # microns2nm * delta(filter) / 2
    wfe_wref = 1000 * 0.5 * (filter_pars['low'] + filter_pars['high'])
    wfe_nm = wfe_wref / 14.0
    rother = r_other(wfe_wref, wfe_nm, params.pm_d)

    return Telescope(
        telescope='Roman OTA',
        instrument_name='Wide-Field Instrument',
        mode='broad band filters',
        sca=sca,
        opt_elem=opt_elem,
        pm_diam=params.pm_d,
        sm_diam=params.sm_d,
        n_refl=params.n_refl,
        n_refr=1,
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
        mode_type=InstrumentMode.ImageOnly,
        dark_cur=params.dark_current,
        frame_time=3.2,
        instrument=ImageInstrument(w_spect=filter_pars['wavelengths']),
    )
