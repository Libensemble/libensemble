"""
libEnsemble.

Library to coordinate the concurrent evaluation of dynamic ensembles of calculations.
"""

from .version import __version__

__author__ = "Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro"
__credits__ = "Argonne National Laboratory"

from libensemble import logger

try:
    from .ensemble import Ensemble
except ModuleNotFoundError:
    pass
