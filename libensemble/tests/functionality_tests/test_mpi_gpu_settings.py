"""
Tests variable resource detection and automatic GPU assignment in libEnsemble.

This test is like regression test test_GPU_variable_resources.py, but uses
the dry_run option to test correct GPU settings for different mocked up systems.
Test assertions are in the sim function via the check_gpu_setting function.

The persistent generator creates simulations with variable resource requirements.

Four GPUs per node is mocked up below (if this line is removed, libEnsemble will
detect any GPUs available).

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

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_GPU_variable_resources as sim_f
from libensemble.gen_funcs.persistent_sampling import uniform_random_sample_with_variable_resources as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.executors.mpi_executor import MPIExecutor

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    # The persistent gen does not need resources

    libE_specs["num_resource_sets"] = nworkers - 1  # Persistent gen does not need resources
    libE_specs["resource_info"] = {"gpus_on_node": 4}  # Mock GPU system / uncomment to detect GPUs

    #TODO Not essential as no app I/O - only reason nice to have is for openmpi machinefiles to be somewhere
    #but when workflow dir is ready - just use that.
    #libE_specs["sim_dirs_make"] = False
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

    for run_set in ["mpich", "openmpi", "aprun", "srun", "jsrun", "custom"]:

        print(f"\nRunning GPU setting checks for {run_set} ------------------- ")
        exctr = MPIExecutor(custom_info={"mpi_runner": run_set})
        exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

        # Reset persis_info. If has num_gens_started > 0 from alloc, will not runs any sims.
        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, _, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

    # All asserts are in sim func

