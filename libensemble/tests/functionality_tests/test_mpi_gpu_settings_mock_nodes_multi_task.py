"""
Tests multi-task (using GPU and non-GPU tasks using variable resource detection
and automatic GPU assignment in libEnsemble.

This test is based on test_mpi_gpu_settings.py.

Uses the dry_run option to test correct GPU settings for different mocked up systems.
Test assertions are in the sim function via the check_gpu_setting function.

The persistent generator creates simulations with variable resource requirements.

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 6 python test_mpi_gpu_settings_mock_nodes_multi_task.py
   python test_mpi_gpu_settings_mock_nodes_multi_task.py --nworkers 5

When running with the above command, the number of concurrent evaluations of
the objective function will be 4, as one of the five workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# The 12 process (11 worker) run is necessary to test configurations that must be adjusted.
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 6 12

import os
import sys

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_diff_simulations as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources_from_gen as sim_f
from libensemble.tests.regression_tests.common import create_node_file
from libensemble.tools import add_unique_random_streams, parse_args

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    nsim_workers = nworkers - 1
    libE_specs["num_resource_sets"] = nsim_workers  # Persistent gen does not need resources
    libE_specs["use_workflow_dir"] = True  # Only a place for Open MPI machinefiles

    if libE_specs["comms"] == "tcp":
        sys.exit("This test only runs with MPI or local -- aborting...")

    # Get paths for applications to run
    six_hump_camel_app = six_hump_camel.__file__

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"dry_run": True},
    }

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "x", "sim_id"],
        "out": [("priority", float), ("num_procs", int), ("num_gpus", int), ("x", float, n)],
        "user": {
            "initial_batch_size": nsim_workers,
            "max_procs": max(nsim_workers // 2, 1),  # Any sim created can req. 1 worker up to max
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "multi_task": True,
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
    exit_criteria = {"sim_max": nsim_workers * 2}

    # Ensure LIBE_PLATFORM environment variable is not set.
    if "LIBE_PLATFORM" in os.environ:
        del os.environ["LIBE_PLATFORM"]

    node_file = "nodelist_mpi_gpu_settings"

    if is_manager:
        create_node_file(num_nodes=2, name=node_file)

    # Mock GPU system / remove to detect GPUs, CPUs, and/or nodes
    libE_specs["resource_info"] = {"gpus_on_node": 4, "node_file": node_file, "cores_on_node": (32, 64)}

    for run_set in ["mpich", "openmpi", "aprun", "srun", "jsrun", "custom"]:
        print(f"\nRunning GPU setting checks (via resource_info / custom_info) for {run_set} ------------- ")
        exctr = MPIExecutor(custom_info={"mpi_runner": run_set})
        exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

        # Reset persis_info. If has num_gens_started > 0 from alloc, will not runs any sims.
        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, _, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

    # Oversubscribe procs
    if nsim_workers >= 4:
        cores_per_node = nsim_workers // 4
    else:
        cores_per_node = 1

    libE_specs["resource_info"] = {
        "gpus_on_node": 4,
        "node_file": node_file,
        "cores_on_node": (cores_per_node, cores_per_node),
    }

    for run_set in ["mpich", "openmpi", "aprun", "srun", "jsrun", "custom"]:
        print(f"\nRunning GPU setting checks (via resource_info / custom_info) for {run_set} ------------- ")
        exctr = MPIExecutor(custom_info={"mpi_runner": run_set})
        exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

        # Reset persis_info. If has num_gens_started > 0 from alloc, will not runs any sims.
        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, _, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

    del libE_specs["resource_info"]

    # All asserts are in sim func
