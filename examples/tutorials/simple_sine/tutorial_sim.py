import numpy as np


def sim_find_sine(H, persis_info, sim_specs, _):
    # underscore for internal/testing arguments

    # Create an output array of a single zero
    out = np.zeros(1, dtype=sim_specs["out"])

    # Set the zero to the sine of the input value stored in H
    out["y"] = np.sin(H["x"])

    # Send back our output and persis_info
    return out, persis_info
