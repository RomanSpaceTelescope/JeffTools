import numpy as np

## Sensor Chip Assembly
# use SCA #5 as default for pixel scale etc. - has intermediate characteristics
# if keyword sca is not defined, or is 0, then use analytic calculation for WFE.
#   (indicate this by setting fov = 0)
# otherwise fov is used to specify which PSF file to read in
# sca_id is used in the parameter structures to indicate which SCA was specified by the user.
# sca_id runs from 1-18
# sca_index is used to specify the pixel scale, focal length, SCA tilt WRT gut ray, etc.
# sca_index specifies which pupil obscuration factor to use - can choose something different later.
# sca_index runs from 0-17


sca_dep_xavg_pixel_scale: np.ndarray = 1e-3 * np.array([
    110.260, 109.448, 108.531, 109.803, 109.027, 108.146, 108.923, 108.238, 107.437,
    110.261, 109.448, 108.531, 109.803, 109.027, 108.147, 108.923, 108.238, 107.437])

sca_dep_yavg_pixel_scale: np.ndarray = 1e-3 * np.array([
    108.182, 105.832, 103.234, 108.353, 106.108, 103.607, 108.764, 106.782, 104.524,
    108.182, 105.832, 103.234, 108.353, 106.108, 103.607, 108.764, 106.782, 104.524])

sca_dep_avg_pixel_scale: np.ndarray = 0.5 * \
    (sca_dep_xavg_pixel_scale + sca_dep_yavg_pixel_scale)

# full - well depth not yet measured as of 2022 06 04, set to requirement as placeholder specified in electrons, not ADU.
sca_dep_full_well = np.array([80000.0, 80000.0, 80000.0, 80000.0, 80000.0, 80000.0,
                              80000.0, 80000.0, 80000.0, 80000.0, 80000.0, 80000.0,
                              80000.0, 80000.0, 80000.0, 80000.0, 80000.0, 80000.0])

# Variation from one SCA to another varies a lot given size of FPA
# Tilt angle of SCA WRT to central ray to that SCA from pupil center
sca_dep_sca_tilt = np.array([12.62556, 16.44526, 19.8444, 12.99057, 16.56297, 19.80865,
                             13.58307, 16.63669, 19.55715, 12.62556, 16.44526, 19.8444,
                             12.99058, 16.56297, 19.80865, 13.58307, 16.63668, 19.55715])
