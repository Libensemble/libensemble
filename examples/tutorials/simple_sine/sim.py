import numpy as np


def sim_find_sine(Input, _, sim_specs):
    # Create an output array of a single zero
    Output = np.zeros(1, dtype=sim_specs["out"])

    # Set the zero to the sine of the Input value
    Output["y"] = np.sin(Input["x"])

    # Send back our output
    return Output
