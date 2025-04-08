"""
Tests variable resource detection and automatic GPU assignment in both
generator and simulators.

The persistent generator creates simulations with variable resource requirements,
while also requiring resources itself. The resources required by a sim must
not be larger than what remains once the generator resources are assigned.

The sim_f (gpu_variable_resources_from_gen) asserts that the GPU assignment
is correct for the default method for the MPI runner. GPUs are not actually
used for the default application. Four GPUs per node is mocked up below (if this line
is removed, libEnsemble will detect any GPUs available).

A dry_run option is provided. This can be set in the calling script, and will
just print run-lines and GPU settings. This may be used for testing run-lines
produced and GPU settings for different MPI runners.

Execute via one of the following commands (e.g., 4 workers):
   mpiexec -np 5 python test_GPU_gen_resources.py
   python test_GPU_gen_resources.py --nworkers 4

When running with the above command, the number of concurrent evaluations of
the objective function will be 4, as one of the five workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 5

import sys

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample_with_sim_gen_resources as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import gpu_variable_resources_from_gen as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

# from libensemble import logger
# logger.set_level("DEBUG")  # For testing the test


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["num_resource_sets"] = nworkers  # Persistent gen DOES need resources

    # Mock GPU system / uncomment to detect GPUs
    libE_specs["sim_dirs_make"] = True  # Will only contain files if dry_run is False
    libE_specs["gen_dirs_make"] = True  # Will only contain files if dry_run is False
    libE_specs["ensemble_dir_path"] = "./ensemble_GPU_gen_resources_w" + str(nworkers)
    libE_specs["reuse_output_dir"] = True
    dry_run = True

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
        "user": {"dry_run": dry_run},
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
            "dry_run": dry_run,
        },
    }

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "give_all_with_same_priority": False,
            "async_return": False,  # False batch returns
        },
    }

    exit_criteria = {"sim_max": 20}
    libE_specs["resource_info"] = {"cores_on_node": (nworkers * 2, nworkers * 4), "gpus_on_node": nworkers}

    base_libE_specs = libE_specs.copy()
    for gen_on_manager in [False, True]:
        for run in range(5):
            # reset
            libE_specs = base_libE_specs.copy()
            libE_specs["gen_on_manager"] = gen_on_manager
            persis_info = add_unique_random_streams({}, nworkers + 1)

            if run == 0:
                libE_specs["gen_num_procs"] = 2
            elif run == 1:
                if gen_on_manager:
                    print("SECOND LIBE CALL WITH GEN ON MANAGER")
                libE_specs["gen_num_gpus"] = 1
            elif run == 2:
                persis_info["gen_num_gpus"] = 1
            elif run == 3:
                # Two GPUs per resource set
                libE_specs["resource_info"]["gpus_on_node"] = nworkers * 2
                persis_info["gen_num_gpus"] = 1
            elif run == 4:
                # Two GPUs requested for gen
                persis_info["gen_num_procs"] = 2
                persis_info["gen_num_gpus"] = 2
                gen_specs["user"]["max_procs"] = max(nworkers - 2, 1)

            # Perform the run
            H, persis_info, flag = libE(
                sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
            )

# All asserts are in gen and sim funcs
