#!/usr/bin/env python
import os
import sys
import numpy as np
from icesheet_simf import run_icesheet  # Sim func from current dir
# from icesheet_simf import run_icesheet, read_stat_file # Sim func from current dir

from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams, save_libE_output
from libensemble.executors import MPIExecutor

from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt as gen_f
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens as alloc_f
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out

# Parse number of workers, comms type, etc. from arguments
nworkers, is_manager, libE_specs, _ = parse_args()

#libE_specs["zero_resource_workers"] = [1]
libE_specs["num_resource_sets"] = nworkers

# Initialize MPI Executor instance
exctr = MPIExecutor()

# Register simulation executable with executor
# sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")
sim_app = os.path.join(os.getcwd(), "/home/kuanghsu.wang/JKS8e4/test/a.out")

# if not os.path.isfile(sim_app):
#     sys.exit("a.out not found - please build first in /home/kuanghsu.wang/libE-IceSheet-main/GPU-code/JKS8e4/ dir")
# #    sys.exit("forces.x not found - please build first in ../forces_app dir")

exctr.register_app(full_path=sim_app, app_name="icesheet")
# exctr.register_app(full_path=sim_app, app_name="forces")

m = 44229

# State the sim_f, inputs, outputs
sim_specs = {
    "sim_f": run_icesheet,  # sim_f, imported above
    "in": ["x"],  # Name of input for sim_f
    # "out": [("f", float), ("fvec", float, 2*m), ("iterations", int), ("error", float)],  # Name, type of output from sim_f, fix velocity_field to error (last error value)
    "out": [("f", float), ("iterations", int), ("error", float)],  # Name, type of output from sim_f, fix velocity_field to error (last error value)
    "user": {"max_size": m},
    }

n = 3
gen_out += [("x", float, n), ("x_on_cube", float, n)]

# State the gen_f, inputs, outputs, additional parameters
gen_specs = {
    "gen_f": gen_f,
    "persis_in": ["x", "f", "sim_id", "iterations"],
    "out": gen_out,
    "user": {
        "lb": np.array([0.9, 0.01, 0.9]),  # User parameters for the gen_f
        "ub": np.array([0.99, 0.05, 0.999]),
        #"lb": np.array([0.9, 0.9, 0.01]),  # User parameters for the gen_f
        #"ub": np.array([0.99, 0.999, 0.05]),
        "gen_batch_size": nworkers - 1,  # Generate one random point x for each of the workers.
        "localopt_method": "LN_BOBYQA",
        "num_active_runs": 1,
        "xtol_rel": 1e-4,
    },
}

alloc_specs = {"alloc_f": alloc_f, "out": gen_out, "user": {"batch_mode": True, "num_active_gens": 1}}

# Create and work inside separate per-simulation directories
libE_specs["sim_dirs_make"] = True

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": 30*n}  # exit_criteria = {"sim_max": 8}

# Seed random streams for each worker, particularly for gen_f
persis_info = add_unique_random_streams({}, nworkers + 1)

# H0 = np.zeros(1, dtype = [('x', float, 3), ('x_on_cube', float, 3), ('f', float), ('fvec', float, 2*m), ('sim_id', int), ('local_pt', bool), ('dist_to_better_l', float), ('dist_to_better_s', float), ('iterations', int), ('error', float) ])
H0 = np.zeros(1, dtype = [('x', float, 3), ('x_on_cube', float, 3), ('f', float), ('sim_id', int), ('local_pt', bool), ('dist_to_better_l', float), ('dist_to_better_s', float), ('iterations', int), ('error', float) ])
H0['x'] = np.array([0.98, 0.03, 0.99])  # The best starting point from random sampling
#H0['x'] = np.array([0.98, 0.99, 0.03])  # The best starting point from random sampling
H0['iterations'] = 1657  # The final iteration for the starting point
H0['x_on_cube'] = (H0['x']-gen_specs["user"]["lb"])/(gen_specs["user"]["ub"]-gen_specs["user"]["lb"])

input_file = "starting_point_error.txt"
#input_file = "error.csv"
#H0['fvec'], _, _ = read_stat_file(input_file, m)
# H0['fvec'], _, _ = read_stat_file(input_file, m)

#H0['fvec'] = np.load("/home/kuanghsu.wang/JKS8e4/test/starting_point_error.txt") # Error for the starting point
#H0['fvec'] = error # Error for the starting point
# H0['f'] = np.sum(H0['fvec']**2)  # Norm of the errors
H0['f'] = H0['iterations'][0] 
#H0['error'] = 3.076126985e-07
H0['x_on_cube'] = (H0['x']-gen_specs["user"]["lb"])/(gen_specs["user"]["ub"]-gen_specs["user"]["lb"])
H0['sim_id'] = 0
H0['dist_to_better_s'] = np.inf
H0['dist_to_better_l'] = np.inf
# Launch libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0 = H0)

if is_manager:
    # print(np.sum(H['fvec']**2,axis=1))
    print(H['f'])
    save_libE_output(H, persis_info, __file__, nworkers)
