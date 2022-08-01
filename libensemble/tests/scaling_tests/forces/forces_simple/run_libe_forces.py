#!/usr/bin/env python
import os
import sys
import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors import MPIExecutor

if __name__ == "__main__":

    # Parse number of workers, comms type, etc. from arguments
    nworkers, is_manager, libE_specs, _ = parse_args()

    # Initialize MPI Executor instance
    exctr = MPIExecutor()

    # Register simulation executable with executor
    sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")

    if not os.path.isfile(sim_app):
        sys.exit("forces.x not found - please build first in ../forces_app dir")

    exctr.register_app(full_path=sim_app, app_name="forces")

    # State the sim_f, inputs, outputs
    sim_specs = {
        "sim_f": run_forces,  # sim_f, imported above
        "in": ["x"],  # Name of input for sim_f
        "out": [("energy", float)],  # Name, type of output from sim_f
    }

    # State the gen_f, inputs, outputs, additional parameters
    gen_specs = {
        "gen_f": uniform_random_sample,  # Generator function
        "in": [],  # Generator input
        "out": [("x", float, (1,))],  # Name, type and size of data from gen_f
        "user": {
            "lb": np.array([1000]),  # User parameters for the gen_f
            "ub": np.array([3000]),
            "gen_batch_size": 8,
        },
    }

    # Create and work inside separate per-simulation directories
    libE_specs["sim_dirs_make"] = True

    # Instruct libEnsemble to exit after this many simulations
    exit_criteria = {"sim_max": 8}

    # Seed random streams for each worker, particularly for gen_f
    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Launch libEnsemble
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)
