import numpy as np


def float_x1000(H, persis_info, sim_specs, _):
    """
    Transforms an array and a scalar input and returns values
    """
    output = np.zeros(1, dtype=sim_specs["out"])

    x1 = H["x"][0][0] * 1000.0
    x2 = H["x"][0][1]

    output["arr_vals"].fill(x1)
    output["scal_val"] = x2 + x2 / 1e7

    return output, persis_info
