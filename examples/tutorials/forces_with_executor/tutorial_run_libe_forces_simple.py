#!/usr/bin/env python
import os
import numpy as np
from tutorial_forces_simf_simple import run_forces  # Sim func from current dir

from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor

nworkers, is_manager, libE_specs, _ = parse_args()  # Convenience function

# Create executor and register sim to it
exctr = MPIExecutor()

# Normally would be pre-compiled
if not os.path.isfile("forces.x"):
    if os.path.isfile("build_forces.sh"):
        import subprocess

        subprocess.check_call(["./build_forces.sh"])

# Register simulation executable with executor
sim_app = os.path.join(os.getcwd(), "forces.x")
exctr.register_app(full_path=sim_app, app_name="forces")

# State the sim_f, its arguments, output, and parameters (and their sizes)
sim_specs = {
    "sim_f": run_forces,  # sim_f, imported above
    "in": ["x"],  # Name of input for sim_f
    "out": [("energy", float)],  # Name, type of output from sim_f
}

# State the gen_f, its arguments, output, and necessary parameters.
gen_specs = {
    "gen_f": uniform_random_sample,  # Generator function
    "in": ["sim_id"],  # Generator input
    "out": [("x", float, (1,))],  # Name, type and size of data from gen_f
    "user": {
        "lb": np.array([1000]),  # User parameters for the gen_f
        "ub": np.array([3000]),
        "gen_batch_size": 8,
    },
}

exit_criteria = {"sim_max": 8}

persis_info = add_unique_random_streams({}, nworkers + 1)

H, persis_info, flag = libE(
    sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs
)
