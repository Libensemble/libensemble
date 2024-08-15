"""
This module contains an example function that evaluates one point of any dimension >=1
"""

__all__ = ["norm_eval"]

import numpy as np

from libensemble.specs import input_fields, output_data


@input_fields(["x"])
@output_data([("f", float)])
def norm_eval(H, persis_info, sim_specs, _):
    """
    Evaluates the vector norm for a single point ``x``.

    .. seealso::
        `test_2d_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_2d_sampling.py>`_ # noqa
    """
    x = H["x"]
    H_o = np.zeros(1, dtype=sim_specs["out"])
    H_o["f"] = np.linalg.norm(x)
    return H_o, persis_info
