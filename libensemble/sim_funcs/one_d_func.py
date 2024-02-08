"""
This module contains an example 1d function
"""

__all__ = ["one_d_example"]

import numpy as np

from libensemble.specs import input_fields, output_data


@input_fields(["x"])
@output_data([("f", float)])
def one_d_example(x, persis_info, sim_specs, _):
    """
    Evaluates the six hump camel function for a single point ``x``.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/functionality_tests/test_fast_alloc.py>`_ # noqa
    """

    H_o = np.zeros(1, dtype=sim_specs["out"])

    H_o["f"] = np.linalg.norm(x)

    return H_o, persis_info
