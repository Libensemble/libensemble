"""
Tests variable resource detection and automatic GPU assignment in libEnsemble

The persistent generator creates simulations with variable resource requirements.

The sim_f (gpu_variable_resources) asserts that GPUs assignment
is correct for the default method for the MPI runner. GPUs are not actually
used for default application. Four GPUs per node is mocked up below (if this line
is removed, libEnsemble will detect any GPUs available).

A dry_run option is provided. This can be set in the calling script, and will
just print run-lines and GPU settings. This may be used for testing run-lines
produced and GPU settings for different MPI runners.

Execute via one of the following commands (e.g. 5 workers):
   mpiexec -np 6 python test_GPU_variable_resources.py
   python test_GPU_variable_resources.py --comms local --nworkers 5

When running with the above command, the number of concurrent evaluations of
the objective function will be 4, as one of the five workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 6

import sys

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_with_procs_gpus as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources_from_gen as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["num_resource_sets"] = nworkers - 1  # Persistent gen does not need resources

    # Mock GPU system / uncomment to detect GPUs
    libE_specs["resource_info"] = {"cores_on_node": (8, 16), "gpus_on_node": 4}

    libE_specs["sim_dirs_make"] = True
    libE_specs["ensemble_dir_path"] = "./ensemble_GPU_variable_w" + str(nworkers)

    if libE_specs["comms"] == "tcp":
        sys.exit("This test only runs with MPI or local -- aborting...")

    # Get paths for applications to run
    six_hump_camel_app = six_hump_camel.__file__
    exctr = MPIExecutor()
    exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"dry_run": False},
    }

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "x", "sim_id"],
        "out": [("num_procs", int), ("num_gpus", int), ("x", float, n)],
        "user": {
            "initial_batch_size": nworkers - 1,
            "max_procs": nworkers - 1,  # Any sim created can req. 1 worker up to all.
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
    exit_criteria = {"sim_max": 40}

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
    )

    if is_manager:
        assert flag == 0
        save_libE_output(H, persis_info, __file__, nworkers)
