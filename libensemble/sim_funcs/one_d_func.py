"""
This module contains an example 1d function
"""
__all__ = ['one_d_example']

import numpy as np


def one_d_example(x, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a single point ``x``.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_
    """

    O = np.zeros(1, dtype=sim_specs['out'])

    O['f'] = np.linalg.norm(x)

    return O, persis_info
