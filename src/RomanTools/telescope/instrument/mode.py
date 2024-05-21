# Mode of Instrument

from enum import IntEnum, auto

__all__ = ['InstrumentMode']


class InstrumentMode(IntEnum):
    ImageOnly = auto()
    GrismSpec = auto()
    PrismSpec = auto()
