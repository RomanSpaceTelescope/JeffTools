from ..roman import RomanInputParams
from ..telescope import Table
from ..tel_elem_table import elements_table

import numpy as np


def upstream_elements(
    rip: RomanInputParams,
    obsc_azimfrac,
) -> Table:
    return elements_table(
        names=['FM1', 'Field Stop', 'FM2', 'TM', 'Pupil Mask'],
        types=['mirror', 'fieldstop', 'mirror', 'mirror', 'pupilstop'],
        materials=[rip.mir_surf, 'none', rip.mir_surf, rip.mir_surf, 'Z307'],
        emissivities=np.array([rip.em_mir, 0.01, rip.em_mir, rip.em_mir, rip.em_z307]),
        temperatures=np.array(
            [rip.fm1_temp, rip.aom_temp, rip.fm2_temp, rip.aom_temp, rip.filter_temp]
        ),
        fnum_outer=np.full(5, rip.aperture_fni),
        fnum_inner=np.array([rip.sfnid, rip.sfnid, rip.sfnid, rip.sfnid, rip.sfnid]),
        tput_factor=np.array(
            [rip.m_tput_fac, 1.0, rip.m_tput_fac, rip.m_tput_fac, 1.0]
        ),
        obsc_azimfrac=np.array([1.0, 1.0, 1.0, 1.0, np.sum(obsc_azimfrac)]),
        th_emis_tput=np.ones(5),
        fdep=np.full(5, True, dtype=bool),
        zone=2,
    )


def downstream_elements(
    rip: RomanInputParams,
) -> Table:
    return elements_table(
        names=['EWA aperture', 'Detector Enclosure', 'Detector'],
        types=['housing', 'housing', 'detector'],
        materials=['Al', 'Z307', 'mct_roman'],
        emissivities=np.array([rip.em_al, rip.em_z307, 0.5]),
        temperatures=np.array(
            [rip.filter_temp, rip.det_housing_temp, rip.det_housing_temp]
        ),
        fnum_outer=np.array([rip.aperture_fno, 0.0, 0.0]),
        fnum_inner=np.array([rip.aperture_fni, rip.fno_coba, 0.0]),
        tput_factor=np.ones(3),
        obsc_azimfrac=np.ones(3),
        th_emis_tput=np.ones(3),
        fdep=np.full(3, False, dtype=bool),
        zone=3,
    )
