"""
This module evaluates the augmented Branin function for multi-fidelity optimization.

Augmented Branin is a modified version of the Branin function with a fidelity parameter.
"""

__all__ = ["augmented_branin", "augmented_branin_func"]

import math
import numpy as np


def augmented_branin(H, persis_info, sim_specs, libE_info):
    """
    Evaluates the augmented Branin function for a collection of points given in ``H["x"]``
    with fidelity values in ``H["fidelity"]``.
    """
    batch = len(H["x"])
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    for i in range(batch):
        x = H["x"][i]
        fidelity = H["fidelity"][i]
        H_o["f"][i] = augmented_branin_func(x.reshape(1, -1), fidelity)[0]

    return H_o, persis_info


def augmented_branin_func(x, fidelity):
    """Augmented Branin function for multi-fidelity optimization."""
    x0 = x[:, 0]
    x1 = x[:, 1]

    t1 = 15 * x1 - (5.1 / (4 * math.pi**2) - 0.1 * (1 - fidelity)) * (15 * x0 - 5) ** 2 + 5 / math.pi * (15 * x0 - 5) - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * np.cos(15 * x0 - 5)
    result = t1**2 + t2 + 10

    return -result  # negate for maximization
