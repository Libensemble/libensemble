"""
libEnsemble.

Library to coordinate the concurrent evaluation of dynamic ensembles of calculations.
"""

__version__ = "0.7.0"
__author__ = 'Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro'
__credits__ = 'Argonne National Laboratory'

from libensemble import libE_logger
from .libE import libE, comms_abort
