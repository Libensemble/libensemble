# Sim_func
__all__ = ["likelihood_calculator"]

import numpy as np


def likelihood_calculator(H, persis_info, sim_specs, _):
    """
    Evaluates likelihood
    """
    H_o = np.zeros(len(H["x"]), dtype=sim_specs["out"])
    for i, x in enumerate(H["x"]):
        H_o["like"][i] = six_hump_camel_func(x)

    return H_o, persis_info, "custom_status"


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    return term1 + term2 + term3
