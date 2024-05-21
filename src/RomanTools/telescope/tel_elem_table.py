from typing import List

from astropy.table import Table
import numpy as np
from numpy.typing import NDArray


__all__ = ['Table', 'elements_table']


def elements_table(
    names: List[str],
    types: List[str],
    materials: List[str],
    emissivities: np.ndarray,
    temperatures: np.ndarray,
    fnum_inner: np.ndarray,
    fnum_outer: np.ndarray,
    tput_factor: np.ndarray,
    obsc_azimfrac: np.ndarray,
    th_emis_tput: np.ndarray,
    fdep: NDArray[np.bool_],
    zone: int,
) -> Table:
    return Table(
        [
            names,
            types,
            materials,
            emissivities,
            temperatures,
            fnum_outer,
            fnum_inner,
            tput_factor,
            obsc_azimfrac,
            th_emis_tput,
            fdep,
            zone,
        ],
        names=(
            'Name',
            'Type',
            'Material',
            'Emissivity',
            'Temperature',
            'Fnum Outer',
            'Fnum Inner',
            'Throughput Factor',
            'Obscuration Azim Frac',
            'Thermal_emission_throughput',
            'Filter Dependent',
            'Thermal Zone',
        ),
    )
