from __future__ import annotations

from . import RomanInputParams
from ..tel_elem_table import elements_table
from .sca_dependent_data import sca_dep_avg_pixel_scale, sca_dep_sca_tilt

import numpy as np
from astropy import units as u
from astropy.units import arcsec

__all__ = ['roman_optics']


def roman_optics(sca: int, params: RomanInputParams) -> dict:
    """
    Constructs and returns a dictionary of optical elements for the Roman telescope.

    This function configures optical elements based on the given sensor chip assembly (SCA)
    index and input parameters, defining their characteristics such as f-numbers, emissivity,
    and temperature.

    Args:
    - sca (int): Index of the sensor chip assembly.
    - params (RomanInputParams): Input parameters containing configurations.

    Returns:
    - dict: A dictionary containing the configured optical elements and additional
            properties such as the system's focal length and pixel scale.

    Raises:
    - KeyError: If an invalid SCA index is provided.

    This setup is necessary for accurate representation of the optical system,
    including primary and secondary mirrors, baffles, and support tubes.
    """

    pixel_scale = sca_dep_avg_pixel_scale[sca]
    pixdim = params.pixel_dimension
    sys_focal_length = (pixdim * 1e-6 * u.rad).to(arcsec).value / pixel_scale

    # ::::::::::::::::: Primary Mirror and Aperture Stop ::::::::::::::::::::: #
    pm_apstop_fnum_outer = sys_focal_length / params.pm_apstop_od
    pm_apstop_fnum_inner = sys_focal_length / params.pm_d
    pm_fnum_outer = pm_apstop_fnum_inner
    pm_fnum_inner = sys_focal_length / params.sm_d

    # :::::::::::::::: Primary Mirror (Cassegrain) Baffle ::::::::::::::::
    # pb = roman_primary_baffle(sys_focal_length)
    pm_baffle1_fnum_inner = sys_focal_length / params.pm_baffle1_id
    pm_baffle1_fnum_outer = sys_focal_length / params.pm_baffle1_od

    # ::::::::::::::::: Secondary Mirror Baffle :::::::::::::::::
    # sb = roman_secondary_baffle(sys_focal_length, sm_d, sm_clear_aperture_d)
    sm_baffle_fnum_inner = sys_focal_length / params.sm_clear_aperture_d
    sm_baffle_fnum_outer = sys_focal_length / params.sm_d

    # ::::::::::::::::: Secondary Mirror Support Tubes (SMSTs) :::::::::::::::::
    # smst = roman_support_tubes(sys_focal_length, pm_d, sm_d, pm_apstop_fnum_inner)
    smst_fnum_inner: float = sys_focal_length / params.sm_d
    smst_fnum_outer: float = pm_apstop_fnum_inner
    smst_zone_fnum_inner: np.ndarray = np.full(3, smst_fnum_inner)
    smst_zone_fnum_outer: np.ndarray = np.full(3, smst_fnum_outer)

    # ::::::::::::::::: Telescope Elements :::::::::::::::::
    tel_elem_names = [
        'PM_AP_STOP',
        'SM_BAFFLE',
        *params.smst_zone_names,
        'PM',
        'SM',
        'PM_INNER_BAFFLE1',
        'PM_INNER_BAFFLE2',
    ]
    tel_elem_types = [
        'pupilstop',
        'pupilstop',
        *params.smst_zone_types,
        'mirror',
        'mirror',
        'pupilstop',
        'pupilstop',
    ]
    tel_elem_materials = [
        'Z307',
        'Z307',
        *params.smst_zone_materials,
        params.mir_surf,
        params.mir_surf,
        'Z307',
        'Z307',
    ]
    tel_elem_emis = np.array(
        [
            params.em_z307,
            params.em_z307,
            *params.smst_zone_emis,
            params.em_mir,
            params.em_mir,
            params.em_z307,
            params.em_z307,
        ]
    )
    tel_elem_temps = np.array(
        [
            params.pm_apstop_temp,
            params.sm_baffle_temp,
            *params.smst_zone_temps,
            params.pm_surf_temp,
            params.sm_surf_temp,
            params.pm_baffle1_temp,
            params.pm_baffle2_temp,
        ]
    )

    tel_elem_fnum_outer = np.array(
        [
            pm_apstop_fnum_outer,
            sm_baffle_fnum_outer,
            *smst_zone_fnum_outer,
            pm_fnum_outer,
            params.sm_fnum_outer,
            pm_baffle1_fnum_outer,
            params.pm_baffle2_fnum_outer,
        ]
    )
    tel_elem_fnum_inner = np.array(
        [
            pm_apstop_fnum_inner,
            sm_baffle_fnum_inner,
            *smst_zone_fnum_inner,
            pm_fnum_inner,
            params.sm_fnum_inner,
            pm_baffle1_fnum_inner,
            params.pm_baffle2_fnum_inner,
        ]
    )

    # assign fudge factors for telescope mirror contamination losses.
    tel_elem_tput_factor = np.array(
        [
            1.0,
            1.0,
            *params.smst_zone_tput_fac,
            params.m_tput_fac,
            params.m_tput_fac,
            1.0,
            1.0,
        ]
    )

    # Define azimuthal non - uniformity of pupil obscuration
    # Differs from unity only for SMSTs, but could be more general if baffles were
    # non-circular, etc.
    # This is used by ips_telescope_thermal_background.pro to compute the fraction
    # of the solid angle encompassed by the inner and outer f/# to use for
    # calculating the thermal emission.
    tel_elem_obsc_azimfrac = np.array(
        [
            1.0,
            1.0,
            *(params.smst_zone_frac_area * params.smst_azim_frac),
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )

    tel_elem_th_emis_tput = np.array(
        [
            params.pm_apstop_th_emis_tput,
            params.sm_baffle_th_emis_tput,
            *params.smst_zone_th_emis_tput,
            params.pm_th_emis_tput,
            params.sm_th_emis_tput,
            params.pm_baffle1_th_emis_tput,
            params.pm_baffle2_th_emis_tput,
        ]
    )

    opt_elem = elements_table(
        names=tel_elem_names,
        types=tel_elem_types,
        materials=tel_elem_materials,
        emissivities=tel_elem_emis,
        temperatures=tel_elem_temps,
        fnum_inner=tel_elem_fnum_inner,
        fnum_outer=tel_elem_fnum_outer,
        tput_factor=tel_elem_tput_factor,
        obsc_azimfrac=tel_elem_obsc_azimfrac,
        th_emis_tput=tel_elem_th_emis_tput,
        fdep=np.full(9, True, dtype=bool),
        zone=1,
    )

    return {
        'pixel_scale': pixel_scale,
        'sys_focal_length': sys_focal_length,
        'tel_elems': opt_elem,
        'focal_length_ratio': sys_focal_length / params.pm_d,
        'cos_sca_tilt': np.cos(np.radians(sca_dep_sca_tilt))[sca],
    }

    # return TelescopeOptics(
    #     tel_elems=opt_elem,
    #     pm_diam=rip.pm_d,
    #     sm_diam=rip.sm_d,
    #     n_refl=rip.n_refl,
    #     sys_focal_length=sys_focal_length,
    #     focal_length_ratio=sys_focal_length / rip.pm_d,
    #     cos_sca_tilt=cos_sca_tilt(sca),
    #     pixel_scale=pixel_scale,
    # )
