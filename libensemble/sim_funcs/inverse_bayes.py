# Sim_func
__all__ = ["likelihood_calculator"]

import numpy as np

from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func


def likelihood_calculator(H, persis_info, sim_specs, _):
    """
    Evaluates likelihood
    """
    H_o = np.zeros(len(H["x"]), dtype=sim_specs["out"])
    for i, x in enumerate(H["x"]):
        H_o["like"][i] = six_hump_camel_func(x)

    return H_o, persis_info, "custom_status"
