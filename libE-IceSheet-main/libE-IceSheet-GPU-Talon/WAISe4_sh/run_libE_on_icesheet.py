#!/usr/bin/env python
import os
import sys
import numpy as np

from icesheet_simf import run_icesheet, read_stat_file  # Sim func from current dir

from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams, save_libE_output
from libensemble.executors import MPIExecutor

from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt as gen_f
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens as alloc_f
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out

# Parse number of workers, comms type, etc. from arguments
nworkers, is_manager, libE_specs, _ = parse_args()

#libE_specs["zero_resource_workers"] = [1]
libE_specs["num_resource_sets"] = nworkers - 1

# Initialize MPI Executor instance
# exctr = MPIExecutor(custom_info={"mpi_runner": "srun"})  # perlmutter
exctr = MPIExecutor()

# Register simulation executable with executor
# sim_app = os.path.join(os.getcwd(), "/global/homes/s/shuds/perlmutter_files/icesheet/libE-IceSheet-GPU-Talon/ssa_fem_pt.x")  #perlmutter
sim_app = os.path.join(os.getcwd(), "/lcrc/project/libE_gpu/IceSheet_models/WAISe4/test/a.out")  #swing
exctr.register_app(full_path=sim_app, app_name="icesheet")

# State the sim_f, inputs, outputs

m = 39687  # WAISe4

sim_specs = {
    "sim_f": run_icesheet,  # sim_f, imported above
    "in": ["x"],  # Name of input for sim_f
    "out": [("iterations", int),  #sh - may not need?
            ("f", float),
            ('fvec', float, 2*m)],
    "user": {"max_size": m},
    }


n = 3
gen_out += [("x", float, n), ("x_on_cube", float, n)]

# State the gen_f, inputs, outputs, additional parameters
gen_specs = {
    "gen_f": gen_f,
    "persis_in": ["x", "f", "sim_id", "fvec"],
    "out": gen_out,
    "user": {
        "lb": np.array([0.65, 0.6, 0.070]),  # User parameters for the gen_f  #Anjalis
        "ub": np.array([0.81, 0.85, 0.09]),
        #"lb": np.array([0.1, 0.01, 0.01]),  # User parameters for the gen_f  # jeffs
        #"ub": np.array([1, 1, 0.1]),

        "gen_batch_size": nworkers - 1,  # Generate one random point x for each of the sim workers.
        "localopt_method": "DFOLS",
        "num_active_runs": 1,
        "xtol_rel": 1e-4,
    },
}

alloc_specs = {"alloc_f": alloc_f, "out": gen_out, "user": {"batch_mode": True, "num_active_gens": 1}}

# Create and work inside separate per-simulation directories
libE_specs["sim_dirs_make"] = True

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": nworkers - 1}  # exit_criteria = {"sim_max": 8}

# Seed random streams for each worker, particularly for gen_f
persis_info = add_unique_random_streams({}, nworkers + 1)

#anjalis
#H0 = np.zeros(1, dtype = [('x', float, 3), ('x_on_cube', float, 3), ('f', int), ('sim_id', int), ('local_pt', bool), ('dist_to_better_l', float), ('dist_to_better_s', float) ])
#H0['x'] = np.array([0.799, 0.65, 0.0808])
#H0['x_on_cube'] = (H0['x']-gen_specs["user"]["lb"])/(gen_specs["user"]["ub"]-gen_specs["user"]["lb"])
#H0['f'] = 100
#H0['sim_id'] = 0
#H0['dist_to_better_s'] = np.inf
#H0['dist_to_better_l'] = np.inf

#jeffs - with mods
H0 = np.zeros(1, dtype = [('x', float, 3), ('x_on_cube', float, 3), ('f', float), ('fvec', float, m*2), ('sim_id', int), ('local_pt', bool), ('dist_to_better_l', float), ('dist_to_better_s', float) ])

#H0['x'] = np.array([0.799, 0.65, 0.0808])  # From Anjalis file
#H0['x'] = np.array([0.2, 0.1, 0.05])  # From jeffs test
H0['x'] = np.array([0.707, 0.748, 0.0722]) #sh - from slack message (could be also read from file)


#H0['fvec'] = np.random.uniform(0,1,m*2)  #sh testing

input_file = "WAIS_vectorfield.txt"
H0['fvec'], _, _ = read_stat_file(input_file, m)


H0['f'] = np.sum(H0['fvec']**2)

H0['x_on_cube'] = (H0['x']-gen_specs["user"]["lb"])/(gen_specs["user"]["ub"]-gen_specs["user"]["lb"])
H0['sim_id'] = 0
H0['dist_to_better_s'] = np.inf
H0['dist_to_better_l'] = np.inf

# Launch libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0 = H0)

if is_manager:
    print(H[['x','f']])
    print(np.sum(H['fvec']**2,axis=1))
    save_libE_output(H, persis_info, __file__, nworkers)

