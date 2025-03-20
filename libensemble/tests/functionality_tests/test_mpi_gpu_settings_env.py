"""
Tests options to run an application in a bash-specified environment, without
affecting the parent environment.

This is based on the variable resource detection and automatic GPU assignment test.

This test uses the dry_run option to test the correct runline and GPU settings
for different mocked-up systems. Test assertions are in the sim function via
the check_mpi_runner and check_gpu_setting functions.

Execute via one of the following commands (e.g., 5 workers):
   mpiexec -np 6 python test_mpi_gpu_settings_env.py
   python test_mpi_gpu_settings_env.py --nworkers 5

When running with the above command, the number of concurrent evaluations of
the objective function will be 4, as one of the five workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 6

import os
import sys

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources_subenv as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["num_resource_sets"] = nworkers - 1  # Persistent gen does not need resources
    libE_specs["use_workflow_dir"] = True  # Only a place for Open MPI machinefiles

    # Optional for organization of output scripts
    libE_specs["sim_dirs_make"] = True

    if libE_specs["comms"] == "tcp":
        sys.exit("This test only runs with MPI or local -- aborting...")

    # Get paths for applications to run
    six_hump_camel_app = six_hump_camel.__file__

    env_script_path = os.path.join(os.getcwd(), "./env_script_in.sh")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"dry_run": True, "env_script": env_script_path},
    }

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "x", "sim_id"],
        "out": [("priority", float), ("resource_sets", int), ("x", float, n)],
        "user": {
            "initial_batch_size": nworkers - 1,
            "max_resource_sets": nworkers - 1,  # Any sim created can req. 1 worker up to all.
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "give_all_with_same_priority": False,
            "async_return": False,  # False batch returns
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)
    exit_criteria = {"sim_max": 10}

    # Ensure LIBE_PLATFORM environment variable is not set.
    if "LIBE_PLATFORM" in os.environ:
        del os.environ["LIBE_PLATFORM"]

    libE_specs["resource_info"] = {"gpus_on_node": 4}  # Mock GPU system / remove to detect GPUs

    exctr = MPIExecutor(custom_info={"mpi_runner": "mpich"})
    exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

    # Perform the run
    H, _, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs)
