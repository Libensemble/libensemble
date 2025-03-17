import numpy as np


def mock_sim(H, persis_info, sim_specs, libE_info):
    """
    Places ``sim_specs["out"]`` from a numpy file into the outputs. Allows a
    user to reproduce an existing run while, for example, capturing additional
    information from a gen. Requires a user to have set
    ``sim_specs["user"]["history_file"]`` to point to a history file from a
    previous run.
    """

    hfile = sim_specs["user"]["history_file"]
    nparray = np.load(hfile)
    row = libE_info["H_rows"][:][0]
    libE_output = np.zeros(1, dtype=sim_specs["out"])

    for field in libE_output.dtype.names:
        libE_output[field] = nparray[row][field]

    return libE_output, persis_info
