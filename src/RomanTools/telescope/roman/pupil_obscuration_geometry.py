import numpy as np

from . import RomanInputParams


def pupil_obscuration_geom(
    params: RomanInputParams,
    expupil_rim_id: float | np.ndarray,
    expupil_center_od: float | np.ndarray,
    expupil_leg_width: float | np.ndarray,
) -> dict:
    """
    Calculates pupil obscuration geometry based on telescope parameters.

    This function computes the effective pupil dimensions and the obscuration due to
    the pupil mask legs. It adjusts for the pupil magnification and determines the
    fractional area obscured by mask components.

    Args:
    - params (RomanInputParams): Parameters including telescope and pupil magnification.
    - expupil_rim_id (float | np.ndarray): ID of the exit pupil's rim.
    - expupil_center_od (float | np.ndarray): OD of the exit pupil's center.
    - expupil_leg_width (float | np.ndarray): Width of the exit pupil's legs.

    Returns:
    - dict: A dictionary with keys describing the geometry and obscuration fractions.
    """
    # set obscuration by larger of entrance or exit pupil item
    # mask_outer_rim_id, mask_center_od, and mask_leg_width were scaled by pupil_mag
    # so comparison against pm_d, sm_d, smst_width is correct.
    mask_outer_rim_id = expupil_rim_id * params.pupil_mag
    mask_center_od = expupil_center_od * params.pupil_mag
    mask_leg_width = expupil_leg_width * params.pupil_mag

    eff_outer_rim_id = np.minimum(params.pm_d, mask_outer_rim_id)
    eff_center_od = np.maximum(params.sm_d, mask_center_od)
    eff_leg_width = np.maximum(params.smst_width, mask_leg_width)

    # compute azimuthal fraction for just the cold pupil mask legs
    # Note that SMSTs do not block any telescope emission from reaching the detector!
    total_mask_leg_area = (
        0.5 * params.n_legs * mask_leg_width * (eff_outer_rim_id - eff_center_od)
    )
    mask_leg_azim_frac = total_mask_leg_area / (
        0.25 * np.pi * (eff_outer_rim_id**2 - eff_center_od**2)
    )

    # properties of cold stop
    # When SCA-dependence is added, we will presumably have a table indicating
    # how much of the entrance pupil is visible from behind the cold mask for
    # each SCA. Then calculation will be to add that fraction of the entrance
    # pupil item to the cold-pupil obscuration.
    eff_leg_area = eff_leg_width * (eff_outer_rim_id - eff_center_od) / 2.0
    total_leg_area = params.n_legs * eff_leg_area
    eff_open_area = (
        0.25 * np.pi * (eff_outer_rim_id**2 - eff_center_od**2) - total_leg_area
    )
    obsc_filt_pupil_geom = (params.max_open_area - eff_open_area) / params.max_open_area

    return {
        'mask_outer_rim_id': mask_outer_rim_id,
        'mask_center_od': mask_center_od,
        'mask_leg_width': mask_leg_width,
        'mask_leg_azim_frac': mask_leg_azim_frac,
        'obsc_filt_pupil_geom': obsc_filt_pupil_geom,
        # 'eff_outer_rim_id': eff_outer_rim_id,
        # 'eff_center_od': eff_center_od,
    }
