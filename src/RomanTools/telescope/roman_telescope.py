from . import RomanInputParams
from .instrument import InstrumentMode
from .roman_grism_telescope import roman_grism_telescope
from .roman_image_telescope import roman_image_telescope
from .roman_prism_telescope import roman_prism_telescope
from .telescope import Telescope

__all__ = ['roman_telescope']


def roman_telescope(
    im_mode: InstrumentMode,
    sca: int = 4,
    roman_input_param: RomanInputParams | None = None,
) -> Telescope:
    if sca < 0 or sca >= 18:
        raise KeyError(f'sca index outside range [0,17]: {sca}')

    params: RomanInputParams = (
        roman_input_param if roman_input_param else RomanInputParams()
    )

    if im_mode == InstrumentMode.ImageOnly:
        return roman_image_telescope(sca, params)
    elif im_mode == InstrumentMode.GrismSpec:
        return roman_grism_telescope(sca, params)
    elif im_mode == InstrumentMode.PrismSpec:
        return roman_prism_telescope(sca, params)
    # else:
    #     raise KeyError(im_mode)
