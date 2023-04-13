"""
Tests variable resource detection and automatic GPU assignment in libEnsemble.

This test is like the regression test test_GPU_variable_resources.py, but uses
the dry_run option to test correct GPU settings for different mocked up systems.
Test assertions are in the sim function via the check_gpu_setting function.

The persistent generator creates simulations with variable resource requirements.

Runs three sets of tests.

Set 1.
Four GPUs per node is mocked up below (if this line is removed, libEnsemble will
detect any GPUs available). MPI runner is provided via Executor custom_info

Set 2.
A platform_spec is used. The MPI runner is changed for each call.

Set 3.
A known platform is specified for known systems.

Set 4.
A known platform is specified for known systems via environment variable.

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 6 python test_mpi_gpu_settings.py
   python test_mpi_gpu_settings.py --comms local --nworkers 5

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

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources as sim_f
from libensemble.gen_funcs.persistent_sampling import uniform_random_sample_with_variable_resources as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["num_resource_sets"] = nworkers - 1  # Persistent gen does not need resources

    #libE_specs["use_workflow_dir"] = True  # Only a place for Open machinefiles

    #TODO Not essential as no app I/O - only reason nice to have is for openmpi machinefiles to be somewhere
    #but when workflow dir is ready - just use that.
    #libE_specs["sim_dirs_make"] = True
    #libE_specs["ensemble_dir_path"] = "./ensemble_mpi_gpus_settings_w" + str(nworkers)

    if libE_specs["comms"] == "tcp":
        sys.exit("This test only runs with MPI or local -- aborting...")

    # Get paths for applications to run
    six_hump_camel_app = six_hump_camel.__file__

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"dry_run": True
            },
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
    exit_criteria = {"sim_max": 20, "wallclock_max": 300}

    # Ensure LIBE_PLATFORM environment variable is not set.
    if "LIBE_PLATFORM" in os.environ:
        del os.environ["LIBE_PLATFORM"]

    # First set - use executor setting ------------------------------------------------------------
    libE_specs["resource_info"] = {"gpus_on_node": 4}  # Mock GPU system / uncomment to detect GPUs

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

    del libE_specs["resource_info"]  # this would override

    # Second set - use platform_spec  setting -----------------------------------------------------
    libE_specs["platform_spec"] = {
            # "mpi_runner" : run_set,  # fill in for each run
            "cores_per_node": 64,
            "logical_cores_per_node": 128,
            "gpus_per_node" : 8,
            "gpu_setting_type": "runner_default",
            "scheduler_match_slots": False,
            }

    for run_set in ["mpich", "openmpi", "aprun", "srun", "jsrun", "custom"]:

        print(f"\nRunning GPU setting checks (via platform_spec) for {run_set} ------------------- ")
        libE_specs["platform_spec"]["mpi_runner"] = run_set

        exctr = MPIExecutor()
        exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

        # check having only cores_per_node
        if run_set == "jsrun":
            del libE_specs["platform_spec"]["logical_cores_per_node"]

        if run_set == "custom":
            del libE_specs["platform_spec"]["cores_per_node"]
            libE_specs["platform_spec"]["logical_cores_per_node"] = 128

        # Reset persis_info. If has num_gens_started > 0 from alloc, will not runs any sims.
        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, _, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

    del libE_specs["platform_spec"]

    # Third set - use platform setting ------------------------------------------------------------
    for platform in ["summit", "crusher", "perlmutter_g", "polaris", "sunspot"]:

        print(f"\nRunning GPU setting checks (via known platform) for {platform} ------------------- ")
        libE_specs["platform"] = platform

        exctr = MPIExecutor()
        exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

        # Reset persis_info. If has num_gens_started > 0 from alloc, will not runs any sims.
        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, _, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

        del libE_specs["platform"]

    # Fourth set - use platform environment setting -----------------------------------------------
    for platform in ["summit", "crusher", "perlmutter_g", "polaris", "sunspot"]:

        print(f"\nRunning GPU setting checks (via known platform env. variable) for {platform} ----- ")
        os.environ["LIBE_PLATFORM"] = platform

        exctr = MPIExecutor()
        exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

        # Reset persis_info. If has num_gens_started > 0 from alloc, will not runs any sims.
        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, _, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

        del os.environ["LIBE_PLATFORM"]

    # All asserts are in sim func
