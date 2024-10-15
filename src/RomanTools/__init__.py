from .telescope import Telescope
from .telescope.instrument.mode import InstrumentMode
from .telescope.roman_input_parameters import RomanInputParams
from .telescope.roman_telescope import roman_telescope

# Expose individual enum elements
ImageOnly = InstrumentMode.ImageOnly
GrismSpec = InstrumentMode.GrismSpec
PrismSpec = InstrumentMode.PrismSpec

__all__ = [
    'InstrumentMode',
    'ImageOnly',
    'GrismSpec',
    'PrismSpec',
    'RomanInputParams',
    'roman_telescope',
    'Telescope',
]
