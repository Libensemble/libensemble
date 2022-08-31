import numpy as np


def six_hump_camel(H, persis_info, sim_specs, _):
    """Six-Hump Camel sim_f."""
    batch = len(H["x"])  # Num evaluations each sim_f call.
    H_o = np.zeros(batch, dtype=sim_specs["out"])  # Define output array H

    for i, x in enumerate(H["x"]):
        H_o["f"][i] = six_hump_camel_func(x)  # Function evaluations placed into H

    return H_o, persis_info


def six_hump_camel_func(x):
    """Six-Hump Camel function definition"""
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    return term1 + term2 + term3
