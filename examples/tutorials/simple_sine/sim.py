import numpy as np


def sim_find_sine(InputArray, _, sim_specs):
    # Create an output array of a single zero
    OutputArray = np.zeros(1, dtype=sim_specs["out"])

    # Set the zero to the sine of the InputArray value
    OutputArray["y"] = np.sin(InputArray["x"])

    # Send back our output
    return OutputArray
