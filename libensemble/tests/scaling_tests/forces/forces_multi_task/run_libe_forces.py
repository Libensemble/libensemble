#!/usr/bin/env python

"""
This example runs two difference applications, one that uses only CPUs and one
that uses GPUs. Both uses a variable number of processors. The GPU application
uses one GPU per processor. As the generator creates simulations, it randomly
assigns between one and max_proc processors to each simulation, and also randomly
assigns which application is to be run.

The forces.c application should be compiled for the CPU to `forces_cpu.x`, and
for the GPU (setting the GPU preprocessor condition) to `forces_gpu.x`.

For compile lines, see examples in ../forces_app/build_forces.sh.
"""

import os
import sys

import numpy as np
from forces_simf import run_forces  # Sim func from current dir

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_diff_simulations as gen_f
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
cpu_app = os.path.join(os.getcwd(), "../forces_app/forces_cpu.x")
gpu_app = os.path.join(os.getcwd(), "../forces_app/forces_gpu.x")

if not os.path.isfile(cpu_app):
    sys.exit(f"{cpu_app} not found - please build first in ../forces_app dir")
if not os.path.isfile(gpu_app):
    sys.exit(f"{gpu_app} not found - please build first in ../forces_app dir")

exctr.register_app(full_path=cpu_app, app_name="cpu_app")
exctr.register_app(full_path=gpu_app, app_name="gpu_app")

# State the sim_f, inputs, outputs
sim_specs = {
    "sim_f": run_forces,  # sim_f, imported above
    "in": ["x", "app_type"],  # Name of input for sim_f
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
        ("app_type", 'S10'),
    ],
    "user": {
        "lb": np.array([5000]),  # fewest particles (changing will change checksum)
        "ub": np.array([10000]),  # max particles (changing will change checksum)
        "initial_batch_size": nsim_workers,
        "max_procs": (nsim_workers) // 2,  # Any sim created can req. 1 worker up to max
        "multi_task": True,
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

# Uncomment to see resource sets in libE_stats.txt
# libE_specs["stats_fmt"] = {"show_resource_sets": True}

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": nsim_workers * 2}

# Seed random streams for each worker, particularly for gen_f
persis_info = add_unique_random_streams({}, nworkers + 1)

# Launch libEnsemble
H, persis_info, flag = libE(
    sim_specs, gen_specs, exit_criteria, persis_info=persis_info, alloc_specs=alloc_specs, libE_specs=libE_specs
)
