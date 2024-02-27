"""
Tests CUDA variable resource detection in libEnsemble

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_sampling_CUDA_variable_resources.py

When running with the above command, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4

import sys

import numpy as np

from libensemble import logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.persistent_sampling_var_resources import uniform_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.var_resources import CUDA_variable_resources as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

logger.set_level("DEBUG")


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    # The persistent gen does not need resources

    libE_specs["num_resource_sets"] = nworkers  # Any worker can be the gen

    # libE_specs["zero_resource_workers"] = [0]  # If first worker must be gen, use this instead

    libE_specs["sim_dirs_make"] = True
    libE_specs["workflow_dir_path"] = "./ensemble_CUDA/workflow_" + libE_specs["comms"] + "_w" + str(nworkers) + "_N"
    libE_specs["sim_dir_copy_files"] = [".gitignore"]
    libE_specs["reuse_output_dir"] = True
    libE_specs["manager_runs_additional_worker"] = True

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
        "user": {},
    }

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "x", "sim_id"],
        "out": [("resource_sets", int), ("x", float, n)],
        "user": {
            "initial_batch_size": nworkers,
            "max_resource_sets": nworkers,  # Any sim created can req. 1 worker up to all.
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "give_all_with_same_priority": False,
            "async_return": True,
        },
    }

    libE_specs["scheduler_opts"] = {"match_slots": True}
    exit_criteria = {"sim_max": 40, "wallclock_max": 300}

    # Perform the run

    for i in range(2):
        persis_info = add_unique_random_streams({}, nworkers + 1)
        libE_specs["workflow_dir_path"] = libE_specs["workflow_dir_path"][:-1] + str(i)
        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

    if is_manager:
        assert flag == 0
        save_libE_output(H, persis_info, __file__, nworkers)
