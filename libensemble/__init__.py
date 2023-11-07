"""
libEnsemble.

Library to coordinate the concurrent evaluation of dynamic ensembles of calculations.
"""

from .version import __version__

__author__ = "Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro"
__credits__ = "Argonne National Laboratory"

from libensemble import logger
from libensemble.utils.pydantic_support import pydanticV1, pydanticV2

if pydanticV1:
    from .specs import specsV1 as specs
elif pydanticV2:
    from .specs import specsV2 as specs

from .ensemble import Ensemble
