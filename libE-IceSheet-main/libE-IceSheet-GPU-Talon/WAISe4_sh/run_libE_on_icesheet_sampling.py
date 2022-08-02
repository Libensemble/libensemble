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
# exctr = MPIExecutor(custom_info={"mpi_runner": "srun"})  # perlmutter
exctr = MPIExecutor()

# Register simulation executable with executor
# sim_app = os.path.join(os.getcwd(), "/global/homes/s/shuds/perlmutter_files/icesheet/libE-IceSheet-GPU-Talon/ssa_fem_pt.x")  # perlmutter
sim_app = os.path.join(os.getcwd(), "/lcrc/project/libE_gpu/IceSheet_models/WAISe4/test/a.out")  #swing
exctr.register_app(full_path=sim_app, app_name="icesheet")

# State the sim_f, inputs, outputs
sim_specs = {
    "sim_f": run_icesheet,  # sim_f, imported above
    "in": ["x"],  # Name of input for sim_f
    "out": [("iterations", int), ("error", float)],   # Name, type of output from sim_f, fix velocity_field to error (last error value)
    "user": {},  # Field expected by sim_f
}
# State the gen_f, inputs, outputs, additional parameters
gen_specs = {
    "gen_f": uniform_random_sample,  # Generator function, ask how to implement BOBYQA. 
    "out": [("x", float, (3,))],  # Name, type and size of data from gen_f (For icesheet, x[0] is damping, x[1] is velocity relaxation factor, x[2] is viscosity relaxation factor.
    "user": {
    "lb": np.array([0.5, 0.01, 0.01]),
    "ub": np.array([0.99, 1, 0.1]),
        "gen_batch_size": nworkers, # Generate one random point x for each of the workers.
    },
}

# Create and work inside separate per-simulation directories
libE_specs["sim_dirs_make"] = True

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": 1000}

# Seed random streams for each worker, particularly for gen_f
persis_info = add_unique_random_streams({}, nworkers + 1, 400)

# Launch libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)

if is_manager:
    save_libE_output(H, persis_info, __file__, nworkers)

