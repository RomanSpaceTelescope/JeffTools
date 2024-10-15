from typing import List

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel


class RomanInputParams(BaseModel):
    """
    Configuration class for defining optical and thermal parameters of the Roman
    Space Telescope's mirrors, baffles, optics, and instrument elements.

    Attributes:
    -----------
    pixel_dimension : float
        Size of detector pixels in microns (default: 10.0).
    n_refl : int
        Number of reflective surfaces in the optical path (default: 5).

    Mirror and Optics Parameters:
    -----------------------------
    em_z306, em_z307, em_ag, em_au, em_mli, em_glass, em_al : float
        Emissivity values for various materials used in the telescope optics.
    mir_surf : str
        Name of the primary mirror's surface material.

    Primary and Secondary Mirrors:
    ------------------------------
    sm_d, sm_clear_aperture_d : float
        Diameters of secondary mirror components.
    sm_surf_temp, pm_surf_temp : float
        Surface temperatures of primary and secondary mirrors.
    pm_d, pm_apstop_od : float
        Diameters of primary mirror and aperture stop.
    pupil_mag : float
        Magnification factor from entrance to exit pupil.

    Baffles and Support Structures:
    -------------------------------
    pm_baffle1_id, pm_baffle1_od : float
        Inner and outer diameters of the first part of the primary mirror baffle.
    pm_baffle2_id, pm_baffle2_od : float
        Diameters for the second part of the primary mirror baffle.
    n_legs, smst_width, smst_area : float
        Parameters defining the support tubes (SMSTs) and their obscuration.

    Instrument Elements:
    --------------------
    aom_temp, fm1_temp, fm2_temp, filter_temp : float
        Temperatures of aft optics and other instrument components.
    m_tput_fac, det_tput_fac : float
        Throughput factors for mirrors and detectors, including losses.

    Focal Plane and Pupil Distances:
    --------------------------------
    pupil_distance, grism_rear_surf_distance : float
        Distance from pupil or rear surface of the grism to the focal plane.

    Aperture Parameters:
    --------------------
    aperture_id, aperture_od : float
        Inner and outer diameters of the aluminum aperture in the filter holder.
    aperture_fno, aperture_fni : float
        F-numbers for the aperture.

    Other Parameters:
    -----------------
    dark_current : float
        Dark current in the detectors (default: 0.005 e-/s/pix).
    filt_tmax : float
        Maximum filter transmission (default: 0.95).
    img_dw : float
        Image spectral resolution step size (default: 0.005 microns).
    max_open_area : float
        Maximum open area of the primary mirror.

    """

    # Pre Defined, independent Scalars
    pixel_dimension: float = 10.0
    n_refl: int = 5

    # ::: Mirror and Optics :::
    # Material Parameters
    em_z306: float = 0.9
    em_z307: float = 0.97  # 0.82 was measured at 1um by JWST program; 0.97 from Ball
    em_ag: float = 0.02  # 0.014 is what is assumed by Scott Rohrbach.
    em_au: float = 0.025
    em_mli: float = 0.1  # made up placeholder
    em_glass: float = 0.01  # not true at longer wavelengths!
    em_al: float = 0.024  # from Ball spreadsheet at > ~2um.
    em_mir: float = em_ag
    mir_surf: str = 'Roman_OTA'

    ############################################################
    # Mirror Optical Elements
    ############################################################

    # ::::::::::::::::: Secondary Mirror ::::::::::::::::::::: #
    # sm_d is diameter of the central obscuration - SM scraper, not cassegrain
    # baffle at PM This defines inside f/# of system unless there is an
    # oversize central mask at the exit pupil 0.358m radius per email from
    # Scott Rohrbach 1/27/2020.  This gives 30.3% linear obscuration.
    # Tabs outer radius of 0.398m, but am bookkeeping that solid angle with the SMSTs.
    sm_d: float = 0.716  # 0.358 * 2.0

    # 0.287m ID on SM baffle per email from Scott Rohrbach 1/27/2020.
    # Only 0.27815mm radius is coated with silver, but ignore the difference
    # here as the high-emissivity baffle is what matters for the thermal
    # emission calculations.
    sm_clear_aperture_d: float = 0.574  # 2.0 * 0.287

    # 2019 07 29 operational high value front surface; 269K on back surface
    sm_surf_temp: float = 265.0
    sm_th_emis_tput: float = 1.0  # assume no cold pupil mask is present
    sm_fnum_inner: float = 1.0e6  # no obstruction on - axis for SM

    # compute sm_fnum_outer using distance to EAP - will ultimately be
    # corrected via thermal_emission_tput_fac
    # Actual solid angle of SM surface is larger than PM (meaning smaller f/#).
    sm_to_eap_dist: float = 3.2758  # from WFIRST-SYS-SPEC-0055 (OMD) Rev B
    sm_fnum_outer: float = sm_to_eap_dist / sm_clear_aperture_d  # around 5.7

    # ::::::::::::::::: Primary Mirror and Aperture Stop ::::::::::::::::::::: #
    # Matches value used by Guangjun in his pupil image files. Very tiny change.
    pm_d: float = 2.363120
    # Surface is slightly colder despite nominal temp of 265.
    pm_surf_temp: float = 264.0
    # All PM thermal emission goes to detector unless cold pupil is oversized.
    pm_th_emis_tput: float = 1.0
    # Somewhat arbitrary, real limit will be rim of pupil mask.
    pm_apstop_od: float = pm_d + 0.2
    pm_apstop_temp: float = 265.0  # 2019 07 29 operational high value.
    # Following item can vary in principle with SCA and filter.
    # Assume oversize exit pupil mask - placeholder.
    pm_apstop_th_emis_tput: float = 1.0
    # exit pupil size here is for the wide-field instrument
    # Note that the pupil magnification is a property of the telescope and is independent
    # of whatever mask is in place.
    # X-Y average - from Joe Howard email updated 2021-10-25 (Post CDR).
    exit_pupil_diam: float = 0.0883226
    # Scaling from entrance pupil to exit pupil.
    pupil_mag: float = pm_d / exit_pupil_diam

    # :::::::::::::::: Primary Mirror (Cassegrain) Baffle ::::::::::::::::
    # There are two parts: the material in front of the PM, and the portion of
    # the collar that fits w/in the central hole in the PM.
    # This is small but much warmer.
    # This has similar conceptual difficulties as the SM baffle.
    # Part 1 is not really being imaged, however rays that leave it towards the
    # SM are following essentially the same path as if they were being imaged
    # by the PM+SM, so compute the f/# using the system focal length.
    pm_baffle1_od: float = 2.0 * 0.352  # from opto-mechanical definitions
    pm_baffle1_id: float = 2.0 * 0.300  # also from OMD, WIM tab, lines 211-213
    # this is maximum - most is closer to 230K - required max is 260.
    pm_baffle1_temp: float = 240.0
    pm_baffle1_th_emis_tput: float = 1.0  # placeholder assuming no pupil mask.

    # the second part of the baffle is not really visible to the SM but has a
    # direct path to the FPA.
    pm_to_fpa_dist: float = 4.634  # m
    pm_baffle2_id: float = pm_baffle1_id
    pm_baffle2_od: float = 2.0 * 0.32766  # From OMD, WIM tab, line 251
    pm_baffle2_fnum_inner: float = pm_to_fpa_dist / pm_baffle2_id
    pm_baffle2_fnum_outer: float = pm_to_fpa_dist / pm_baffle2_od
    pm_baffle2_temp: float = 260.0  # 2019 07 29 operational high value
    # Following quantity depends on filter and SCA number!
    # assume no cold stop w/central obsc. as a placeholder.
    pm_baffle2_th_emis_tput: float = 1.0

    # ::::::::::::::::: Secondary Mirror Baffle :::::::::::::::::
    # SM baffles are conceptually difficult:
    # Direct path to detector, not imaged by PM, involves angles likely blocked
    # by the cold pupil mask.
    # In this path, the SM baffle is not at a pupil and doesn't have an image
    # at the cold pupil.
    # Indirect path, imaged by PM, has a much smaller solid angle.
    # Use that as placeholder here.
    sm_baffle_temp: float = 265.0  # Scott Rohrbach email 2/28/19
    sm_baffle_th_emis_tput: float = 1.0  # assume no cold pupil mask as placeholder.

    # ::::::::::::::::: Secondary Mirror Support Tubes (SMSTs) :::::::::::::::::
    n_legs: float = 6.0
    smst_width: float = 0.076
    smst_length: float = (pm_d - sm_d) / 2.0  # projected length, not actual length!
    smst_area: float = n_legs * smst_width * smst_length
    # compute obscuration fraction as fraction of total PM clear area
    smst_total_obsc_frac: float = smst_area / (0.25 * np.pi * (pm_d**2))
    # compute fraction between inner and outer f/numbers
    smst_azim_frac: float = smst_area / (0.25 * np.pi * (pm_d**2 - sm_d**2))

    # Following quantity depends on filter and SCA number!
    smst_th_emis_tput: float = 1.0  # assume no mask at exit pupil as placeholder

    # split SMST into multiple thermal zones
    # Looking at presentation from 2019-03-04, the SMST scrapers are split into two outer long sections
    # and an inner gap that is open to the MLI.
    # Lengthwise, there are 24 'long' thermal zone segments and 4 narrow segments which total in area to one long segment.
    # For a nominal SMST temperature of 269K, 8 longitudinal segments are at 241K, 16 are at ~233K, and one is at ~260K.
    smst_temp: float = 269.0
    smst_zone_names: List = ['SMST_cold_zone', 'SMST_med_zone', 'SMST_warm_zone']
    smst_zone_types: List = ['pupilstop', 'pupilstop', 'pupilstop']
    smst_zone_materials: List = ['Z307+MLI', 'Z307+MLI', 'Z307+MLI']
    temp_269: List = np.array([233.0, 241.0, 260.0]).tolist()
    # middle temp in this case is a guesstimate. Probably good enough
    temp_293: List = np.array([236.0, 248.0, 275.0]).tolist()
    tslope: List = (
        (np.array(temp_293) - np.array(temp_269)) / (293.0 - 269.0)
    ).tolist()
    smst_zone_temps: List = (
        np.array(temp_269) + np.array(tslope) * (smst_temp - 269.0)
    ).tolist()
    smst_zone_frac_area: List = np.array([0.64, 0.32, 0.04]).tolist()
    smst_zone_emis: List = np.full(3, 0.66 * em_z307 + 0.34 * em_mli).tolist()
    # Following quantity depends on filter and SCA number!
    smst_zone_th_emis_tput: List = np.full(3, smst_th_emis_tput).tolist()
    smst_zone_tput_fac: List = np.ones(3).tolist()

    # ::::::::::::::::: Instrument Elements ::::::::::::::::::::: #

    # Aft optics module now has a range of temperatures. The generic AOM temperature applies to the WFC tertiary mirror.
    # These temperatures from PSR 2017 12.
    aom_temp: float = 220.0  # update from march 2019 - need to get final value
    fm1_temp: float = aom_temp + 1.0
    fm2_temp: float = aom_temp - 6.0
    filter_temp: float = 188.0
    det_housing_temp: float = 140.0

    # throughput factors - e.g. 3% loss overall for contamination, 10% det QE margin
    # allocate 3% contamination losses evenly across 5 mirrors
    m_tput_fac: float = 0.97**0.2
    det_tput_fac: float = 1.0

    # compute f/# of COBA entrance aperture - estimated from Ball viewfactor spreadsheet: need to get accurate dimensions
    coba_hole_distance: float = 0.57
    coba_hole_radius: float = 0.1221
    fno_coba: float = coba_hole_distance / (2.0 * coba_hole_radius)

    # compute distance from pupil to focal plane using values from WFIRST-SYS-SPEC-0055 Rev B (OMD)
    pupil_distance: float = 1e-3 * np.sqrt(
        (866.958 - 526.606) ** 2
        + (-1501.616 + 912.108) ** 2
        + (-407.505 + 541.097) ** 2
    )  # 0.69369

    grism_rear_surf_distance: float = 1e-3 * np.sqrt(
        (866.958 - 558.424) ** 2
        + (-1501.616 + 967.219) ** 2
        + (-407.505 + 528.608) ** 2
    )  # 0.62884

    # Aluminum "aperture" built in to filter holder
    # Assumed to be up against rear surface of filter - use grism rear surface as worst case
    # dimensions taken from estimate computed from Ball view-factors
    aperture_dist_to_fpa: float = grism_rear_surf_distance
    aperture_id: float = 2.0 * 0.05127
    aperture_od: float = 2.0 * 0.05798
    aperture_fno: float = aperture_dist_to_fpa / aperture_od
    aperture_fni: float = aperture_dist_to_fpa / aperture_id

    # default outer f/# before knowing dimensions of pupil stop is aperture_fni
    # default is no inner mask
    sfnid: float = 1.0e5

    dark_current: float = 0.005  # doing better with new detectors
    filt_tmax: float = 0.95
    img_dw: float = 0.005

    max_open_area: float = 0.25 * np.pi * pm_d**2
