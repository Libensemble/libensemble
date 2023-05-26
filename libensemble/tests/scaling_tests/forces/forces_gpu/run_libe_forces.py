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
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.libE import libE
from libensemble.tools import add_unique_random_streams, parse_args

# Uncomment for var resources
# from libensemble.gen_funcs.sampling import uniform_random_sample_with_variable_resources as gen_f


# Uncomment for var resources (checksum will change due to rng differences)
# from libensemble.gen_funcs.sampling import uniform_random_sample_with_variable_resources as gen_f


# Parse number of workers, comms type, etc. from arguments
nworkers, is_manager, libE_specs, _ = parse_args()

# To test on system without GPUs - compile forces without -DGPU and mock GPUs with this line.
# libE_specs["resource_info"] = {"gpus_on_node": 4}

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
    "out": [
        ("x", float, (1,)),  # Name, type and size of data from gen_f
        # ("resource_sets", int)  # Uncomment for var resources
    ],
    "user": {
        "lb": np.array([50000]),  # fewest particles (changing will change checksum)
        "ub": np.array([100000]),  # max particles (changing will change checksum)
        "gen_batch_size": 8,
        # "max_resource_sets": nworkers  # Uncomment for var resources
    },
}

# Create and work inside separate per-simulation directories
libE_specs["sim_dirs_make"] = True

# Uncomment to see resource sets in libE_stats.txt - useful with var resources
# libE_specs["stats_fmt"] = {"show_resource_sets": True}

# Instruct libEnsemble to exit after this many simulations
exit_criteria = {"sim_max": 8}  # changing will change checksum

# Seed random streams for each worker, particularly for gen_f
persis_info = add_unique_random_streams({}, nworkers + 1)

# Launch libEnsemble
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)

# This is for configuration of this test (inc. lb/ub and sim_max values)
if is_manager:
    if exit_criteria["sim_max"] == 8:
        chksum = np.sum(H["energy"])
        assert np.isclose(chksum, 96288744.35136001), f"energy check sum is {chksum}"
        print("Checksum passed")
    else:
        print("Run complete. A checksum has not been provided for the given sim_max")
