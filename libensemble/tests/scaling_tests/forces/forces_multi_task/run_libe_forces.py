#!/usr/bin/env python

"""
This example is based on the simple forces test. The default number of
particles is increased considerably to give perceptible time on the GPUs when
live-checking GPU usage.

The forces.c application should be built by setting the GPU preprocessor condition
in addition to openMP GPU flags for the given system. See examples in
../forces_app/build_forces.sh. We recommend running forces.x standalone first
and confirm it is running on the GPU (this is given clearly in the output).

An alternative variable resource generator is available (search 'var resources'
in this script and uncomment relevant lines).
"""

import os
import sys

import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble.executors import MPIExecutor

# Fixed resources (one resource set per worker)
# from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_diff_simulations as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams, parse_args

# Parse number of workers, comms type, etc. from arguments
nworkers, is_manager, libE_specs, _ = parse_args()

nsim_workers = nworkers - 1
libE_specs["num_resource_sets"] = nsim_workers  # Persistent gen does not need resources

# To test on system without GPUs - compile forces without -DGPU and mock GPUs with this line.
# libE_specs["resource_info"] = {"gpus_on_node": 4}

# Initialize MPI Executor instance
exctr = MPIExecutor()

# Register simulation executable with executor
sim_app1 = os.path.join(os.getcwd(), "../forces_app/forces_cpu.x")
sim_app2 = os.path.join(os.getcwd(), "../forces_app/forces_gpu.x")

if not os.path.isfile(sim_app1):
    sys.exit(f"{sim_app1} not found - please build first in ../forces_app dir")
if not os.path.isfile(sim_app2):
    sys.exit(f"{sim_app2} not found - please build first in ../forces_app dir")

exctr.register_app(full_path=sim_app1, app_name="forces_cpu")
exctr.register_app(full_path=sim_app2, app_name="forces_gpu")

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
    "out": [
        ("x", float, (1,)),  # Name, type and size of data from gen_f
        ("num_procs", int),
        ("num_gpus", int),
    ],
    "user": {
        "lb": np.array([5000]),  # fewest particles (changing will change checksum)
        "ub": np.array([10000]),  # max particles (changing will change checksum)
        "initial_batch_size": nsim_workers,
        "max_procs": (nsim_workers) // 2,  # Any sim created can req. 1 worker up to max
        "multi_task": True,
        # "max_resource_sets": nworkers  # Uncomment for var resources
    },
}

alloc_specs = {
    "alloc_f": alloc_f,
    "user": {
        "give_all_with_same_priority": False,
        "async_return": False,  # False causes batch returns
    },
}

# Create and work inside separate per-simulation directories
libE_specs["sim_dirs_make"] = True

# Uncomment to see resource sets in libE_stats.txt - useful with var resources
# libE_specs["stats_fmt"] = {"show_resource_sets": True}

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": nsim_workers * 2}  # changing will change checksum

# Seed random streams for each worker, particularly for gen_f
persis_info = add_unique_random_streams({}, nworkers + 1)

# Launch libEnsemble
H, persis_info, flag = libE(
    sim_specs, gen_specs, exit_criteria, persis_info=persis_info, alloc_specs=alloc_specs, libE_specs=libE_specs
)
