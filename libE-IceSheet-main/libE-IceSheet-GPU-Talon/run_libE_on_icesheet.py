#!/usr/bin/env python
import os
import sys
import numpy as np
from icesheet_simf import run_icesheet  # Sim func from current dir

from libensemble.libE import libE
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.tools import parse_args, add_unique_random_streams, save_libE_output
from libensemble.executors import MPIExecutor

# Parse number of workers, comms type, etc. from arguments
nworkers, is_manager, libE_specs, _ = parse_args()

# Initialize MPI Executor instance
exctr = MPIExecutor()

# Register simulation executable with executor
#sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")
sim_app = os.path.join(os.getcwd(), "/home/jchegwidden/GPU-code/JKS8e4/a.out")

# if not os.path.isfile(sim_app):
#     sys.exit("a.out not found - please build first in /home/kuanghsu.wang/libE-IceSheet-main/GPU-code/JKS8e4/ dir")
# #    sys.exit("forces.x not found - please build first in ../forces_app dir")

exctr.register_app(full_path=sim_app, app_name="icesheet")
#exctr.register_app(full_path=sim_app, app_name="forces")

# State the sim_f, inputs, outputs
sim_specs = {
    "sim_f": run_icesheet,  # sim_f, imported above
    "in": ["x"],  # Name of input for sim_f
    "out": [("f", int)],   # Name, type of output from sim_f, fix velocity_field to error (last error value)
}

# State the gen_f, inputs, outputs, additional parameters
gen_specs = {
    "gen_f": uniform_random_sample,  # Generator function
    "in": ["sim_id"],  # Generator input
    "out": [("x", float, (3,))],  # Name, type and size of data from gen_f (For icesheet, x[0] is damping, x[1] is velocity relaxation factor, x[2] is viscosity relaxation factor.
    "user": {
        "lb": np.array([0.1, 0.01, 0.01]),  # User parameters for the gen_f
        "ub": np.array([1, 1, 0.1]),
        "gen_batch_size": nworkers, # Generate one random point x for each of the workers.
    },
}

# Create and work inside separate per-simulation directories
libE_specs["sim_dirs_make"] = True

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": nworkers} # exit_criteria = {"sim_max": 8}

# Seed random streams for each worker, particularly for gen_f
persis_info = add_unique_random_streams({}, nworkers + 1)

# Launch libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)

if is_manager:
    save_libE_output(H, persis_info, __file__, nworkers)

