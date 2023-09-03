#!/usr/bin/env python
import os
import sys

import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor

from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams, parse_args

if __name__ == "__main__":
    # Parse number of workers, comms type, etc. from arguments
    nworkers, is_manager, libE_specs, _ = parse_args()
    nsim_workers = nworkers - 1  # One worker is for persistent generator
    libE_specs["num_resource_sets"] = nsim_workers  # Persistent gen does not need resources

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
        "gen_f": gen_f,  # Generator function
        "in": [],  # Generator input
        "persis_in": ["sim_id"],  # Just send something back to gen to get number of new points.
        "out": [("x", float, (1,))],  # Name, type and size of data from gen_f
        "user": {
            "lb": np.array([1000]),  # min particles
            "ub": np.array([3000]),  # max particles
            "initial_batch_size": nsim_workers,
        },
    }

    # Starts one persistent generator. Simulated values are returned in batch.
    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "async_return": False,  # False causes batch returns
        },
    }

    # Create and work inside separate per-simulation directories
    libE_specs["sim_dirs_make"] = True

    # Instruct libEnsemble to exit after this many simulations
    exit_criteria = {"sim_max": 8}  # Hint: Use nsim_workers*2 to vary with worker count

    # Seed random streams for each worker, particularly for gen_f
    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Launch libEnsemble
    H, persis_info, flag = libE(
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info=persis_info,
        alloc_specs=alloc_specs,
        libE_specs=libE_specs,
    )

if is_manager:
    # Note, this will change if change sim_max, nworkers, lb/ub etc...
    print(f'Final energy checksum: {np.sum(H["energy"])}')
