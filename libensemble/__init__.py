"""
libEnsemble.

Library to coordinate the concurrent evaluation of dynamic ensembles of calculations.
"""

with open("VERSION", "r") as f:
    version_str = f.read().strip('\n')

__version__ = version_str
__author__ = 'Jeffrey Larson, Stephen Hudson, Stefan M. Wild, David Bindel and John-Luke Navarro'
__credits__ = 'Argonne National Laboratory'

from libensemble import libE_logger
from .libE import libE, comms_abort
