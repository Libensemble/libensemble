"""
libEnsemble.

Library to coordinate the concurrent evaluation of dynamic ensembles of calculations.
"""

from .version import __version__

__author__ = "Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro"
__credits__ = "Argonne National Laboratory"

import pydantic

from libensemble import logger

if pydantic.__version__[0] == "1":
    from .specs import specsV1 as specs
elif pydantic.__version__[0] == "2":
    from .specs import specsV2 as specs

from .ensemble import Ensemble
