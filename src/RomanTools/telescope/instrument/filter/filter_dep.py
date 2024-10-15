from typing import Tuple

import numpy as np

##
from astropy.table import Table

# from here down have to be consistent about which distances are used to compute f/#s.
# Need f/# passed by rim of mask for computing solid angles subtended by beam through most of the optics
# pupil_mag is ratio of pm_diam to optical exit pupil diameter as given in the OMD.
# This is not numerically identical to the ratio of the system focal length to pupil_distance.
# The former is ~27 and the latter is ~30.
# Define zone1 as upstream of OTA FM1 - scale f/#s so exit pupil diameter matches pm diam.
# Define zone2 as FM1 up to pupil mask - use f/#s defined by pupil mask and pupil distance to FPA
# Define zone3 as downstream of pupil mask - use f/#s as defined already


def fnum_outer(sys_focal_length, nf, params, expup_rim_id, opt_elem) -> np.ndarray:
    """
    Calculates the filter-dependent outer F-number for different telescope zones.

    This function computes the outer F-numbers for different zones of a telescope system,
    considering system focal length, pupil diameters, and optical element zones. It returns
    an array of F-numbers adjusted per filter and zone based on optical requirements.

    Args:
    - sys_focal_length (float): System focal length.
    - nf (int): Number of filters.
    - params (object): Parameters including pupil and mirror diameters.
    - expup_rim_id (float): Exit pupil rim inner diameter.
    - opt_elem (array): Array containing optical elements and their zones.

    Returns:
    - np.array: Array of calculated outer F-numbers per filter and zone.
    """

    fdep_fnum_outer_z1 = (
        (sys_focal_length / params.pm_d) * params.exit_pupil_diam / expup_rim_id
    )
    fdep_fnum_outer_z2 = params.pupil_distance / expup_rim_id

    idx_z1 = np.where(opt_elem['Thermal Zone'] == 1)[0]
    idx_z2 = np.where(opt_elem['Thermal Zone'] == 2)[0]
    idx_z3 = np.where(opt_elem['Thermal Zone'] == 3)[0]

    #####################################
    # Filter Dependent Fnum Outer array #
    #####################################
    fdep_fnum_outer = np.empty((nf, len(opt_elem)))
    fdep_fnum_outer[:, idx_z1] = np.minimum(
        fdep_fnum_outer_z1[:, None],
        opt_elem[opt_elem['Thermal Zone'] == 1]['Fnum Outer'][None, :],
    )
    fdep_fnum_outer[:, idx_z2] = np.maximum(
        fdep_fnum_outer_z2[:, None],
        opt_elem[opt_elem['Thermal Zone'] == 2]['Fnum Outer'][None, :],
    )
    fdep_fnum_outer[:, idx_z3] = opt_elem[opt_elem['Thermal Zone'] == 3]['Fnum Outer']

    return fdep_fnum_outer


def fnum_inner(sys_focal_length, nf, params, expup_center_od, opt_elem) -> np.ndarray:
    """
    Calculates the filter-dependent inner F-number for different telescope zones.

    This function determines the inner F-numbers for various zones based on system focal
    length, exit pupil diameters, and the specific zones of optical elements. It accounts
    for different scenarios where optical paths might differ, especially considering
    pupil masks.

    Args:
    - sys_focal_length (float): The system's focal length.
    - nf (int): Number of filters.
    - params (object): Parameters including pupil diameters and distances.
    - expup_center_od (float): Exit pupil center outer diameter.
    - opt_elem (array): Array containing optical elements and their properties.

    Returns:
    - np.array: Calculated inner F-numbers for each zone and filter configuration.
    """
    fdep_fnum_inner_z1 = np.full_like(expup_center_od, 1e6)
    fdep_fnum_inner_z2 = np.full_like(expup_center_od, 1e6)

    fdep_fnum_inner_z1[expup_center_od > 0] = (
        (sys_focal_length / params.pm_d)
        * params.exit_pupil_diam
        / expup_center_od[expup_center_od > 0]
    )
    fdep_fnum_inner_z2[expup_center_od > 0] = (
        params.exit_pupil_diam / expup_center_od[expup_center_od > 0]
    )

    idx_z1 = np.where(opt_elem['Thermal Zone'] == 1)[0]
    idx_z2 = np.where(opt_elem['Thermal Zone'] == 2)[0]
    idx_z3 = np.where(opt_elem['Thermal Zone'] == 3)[0]
    idx_pupil = np.where(opt_elem['Name'] == 'Pupil Mask')[0]

    #####################################
    # Filter Dependent Fnum Inner array #
    #####################################
    fdep_fnum_inner = np.empty((nf, len(opt_elem)))
    fdep_fnum_inner[:, idx_z1] = np.minimum(
        fdep_fnum_inner_z1[:, None],
        opt_elem[opt_elem['Thermal Zone'] == 1]['Fnum Inner'][None, :],
    )
    fdep_fnum_inner[:, idx_z2] = np.maximum(
        fdep_fnum_inner_z2[:, None],
        opt_elem[opt_elem['Thermal Zone'] == 2]['Fnum Inner'][None, :],
    )
    # Special case: Pupil Mask
    fdep_fnum_inner[:, idx_pupil] = fdep_fnum_inner_z2.reshape(-1, 1)
    fdep_fnum_inner[:, idx_z3] = opt_elem[opt_elem['Thermal Zone'] == 3]['Fnum Outer']

    return fdep_fnum_inner


def solid_angle(
    nf,
    params,
    expup_rim_od,
    expup_rim_id,
    expup_center_od,
    expup_leg_width,
    opt_elem,
    fdep_fnum_outer,
    fdep_fnum_inner,
    mask_leg_azim_frac,
    mask_leg_width,
) -> np.ndarray:
    """
    Calculates the solid angle subtended by telescope optical elements.

    This function computes the solid angle based on the optical configuration
    which involves different zones of the telescope and their respective geometries.
    It factors in the complexities of how different parts like the pupil mask and
    SMST zones contribute to the overall solid angle.

    Args:
    - nf (int): Number of filters.
    - params (object): System parameters with geometrical data.
    - expup_rim_od/id, expup_center_od, expup_leg_width (float): Exit pupil dimensions.
    - opt_elem (np.ndarray): Optical element data including zones.
    - fdep_fnum_outer/inner (np.ndarray): Dependent outer/inner F-numbers.
    - mask_leg_azim_frac, mask_leg_width (float): Mask leg azimuthal fraction and width.

    Returns:
    - np.ndarray: Array containing calculated solid angles for each filter and element.
    """
    idx_z1 = np.where(opt_elem['Thermal Zone'] == 1)[0]
    idx_z2 = np.where(opt_elem['Thermal Zone'] == 2)[0]
    idx_z3 = np.where(opt_elem['Thermal Zone'] == 3)[0]
    idx_pupil = np.where(opt_elem['Name'] == 'Pupil Mask')[0]
    idx_smst = np.where(
        np.isin(opt_elem['Name'], ['SMST_cold_zone', 'SMST_med_zone', 'SMST_warm_zone'])
    )[0]

    # Filter Dependent Solid Angle
    # beta_min is minimum angle of rays wrt optic axis for each zone
    # beta_max is maximum angle of rays wrt optic axis for each zone
    # shape of cosbmin is (nf, nopt) == (8, 18)
    valid_cosb_inner = np.zeros_like(fdep_fnum_inner)
    valid_cosb_outer = np.zeros_like(fdep_fnum_outer)

    valid_cosb_inner[fdep_fnum_inner != 0] = 1.0 / np.sqrt(
        1.0 + 1.0 / (4.0 * fdep_fnum_inner[fdep_fnum_inner != 0] ** 2)
    )
    valid_cosb_outer[fdep_fnum_outer != 0] = 1.0 / np.sqrt(
        1.0 + 1.0 / (4.0 * fdep_fnum_outer[fdep_fnum_outer != 0] ** 2)
    )

    cosbmin = np.where(fdep_fnum_inner > 2000.0, 1.0, valid_cosb_inner)
    cosbmax = np.where(fdep_fnum_outer < 1e-4, 0.0, valid_cosb_outer)
    cosbarc = 2.0 * np.pi * (cosbmin - cosbmax)

    ##############################
    # Fnum Dependent Solid Angle #
    ##############################
    fdep_solid_angle = np.empty((nf, len(opt_elem)))
    fdep_solid_angle[:, idx_z1] = cosbarc[:, idx_z1] * (
        1.0 - mask_leg_azim_frac
    ).reshape(-1, 1)
    fdep_solid_angle[:, idx_z2] = cosbarc[:, idx_z2] * (
        1.0 - mask_leg_azim_frac
    ).reshape(-1, 1)
    fdep_solid_angle[:, idx_z3] = cosbarc[:, idx_z3]

    ##############################
    # Solid angle scaling dependent upon Pupil Obscuration, filter, optical segment
    ##############################
    # No SMST leg Obscuration
    legless = mask_leg_width <= 0
    if np.any(legless):
        idx = np.where(np.outer(legless, idx_smst))
        fdep_solid_angle[idx] = (
            cosbarc[:, idx_smst] * opt_elem['Obscuration Azim Frac'][idx_smst]
        ).flatten()

    # Big SMST leg Obscuration
    big_leg = mask_leg_width > params.smst_width
    if np.any(big_leg):
        idx = np.where(np.outer(legless, idx_smst))
        fdep_solid_angle[idx] = 0.0

    # Normal SMST leg Obscuration
    normal_leg = ~legless & ~big_leg
    if np.any(normal_leg):
        idx = np.where(np.outer(normal_leg, idx_smst))
        fdep_solid_angle[idx] = (
            (1.0 - mask_leg_width[normal_leg, None] / params.smst_width)
            * cosbarc[normal_leg][:, idx_smst]
            * opt_elem['Obscuration Azim Frac'][idx_smst]
        ).flatten()

    # Final Scalings
    area_rim = 0.25 * np.pi * (expup_rim_od**2 - expup_rim_id**2)
    area_center = 0.25 * np.pi * (expup_center_od**2)
    area_legs = 0.5 * params.n_legs * expup_leg_width * (expup_rim_id - expup_center_od)
    total_area = area_rim + area_center + area_legs
    fdep_solid_angle[:, idx_pupil] = (total_area / params.pupil_distance**2)[:, None]

    return fdep_solid_angle


def fdep_trio(
    params,
    sys_focal_length,
    opt_elem,
    nf,
    filter_names,
    expupil_rim_od,
    expupil_rim_id,
    expupil_center_od,
    expupil_leg_width,
    mask_leg_azim_frac,
    mask_leg_width,
) -> Tuple[Table, Table, Table]:
    """
    Generates tables of dependent F-numbers and solid angles for telescope optics.

    This function calculates filter-dependent F-numbers for both outer and inner zones
    and the solid angles based on the optical element configurations and system parameters.
    The results are stored in Astropy Tables for each optical element across different filters.

    Args:
    - params (object): Telescope system parameters.
    - sys_focal_length (float): System focal length.
    - opt_elem (array): Array of optical elements including their zones.
    - nf (int): Number of filters.
    - filter_names (list): Names of filters.
    - expupil_rim_od/id, expupil_center_od, expupil_leg_width (float): Geometrical parameters.
    - mask_leg_azim_frac, mask_leg_width (float): Mask dimensions affecting the calculations.

    Returns:
    - Tuple[Table, Table, Table]: Tables of outer F-numbers, inner F-numbers, and solid angles.
    """

    fdep_fnum_outer = fnum_outer(sys_focal_length, nf, params, expupil_rim_id, opt_elem)
    fdep_fnum_inner = fnum_inner(
        sys_focal_length, nf, params, expupil_center_od, opt_elem
    )
    fdep_solid_angle = solid_angle(
        nf,
        params,
        expupil_rim_od,
        expupil_rim_id,
        expupil_center_od,
        expupil_leg_width,
        opt_elem,
        fdep_fnum_outer,
        fdep_fnum_inner,
        mask_leg_azim_frac,
        mask_leg_width,
    )

    # Store as Astropy Tables.
    fdep_fnum_outer_tab = Table(fdep_fnum_outer, names=opt_elem['Name'])
    fdep_fnum_outer_tab.add_column(filter_names, name='Filter Name', index=0)
    fdep_fnum_inner_tab = Table(fdep_fnum_inner, names=opt_elem['Name'])
    fdep_fnum_inner_tab.add_column(filter_names, name='Filter Name', index=0)
    fdep_solid_angle_tab = Table(fdep_solid_angle, names=opt_elem['Name'])
    fdep_solid_angle_tab.add_column(filter_names, name='Filter Name', index=0)

    return (
        fdep_fnum_outer_tab,
        fdep_fnum_inner_tab,
        fdep_solid_angle_tab,
    )
