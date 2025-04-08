"""
Runs libEnsemble doing uniform sampling and then evaluates those points with
varying amount of resources.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_uniform_sampling_with_variable_resources.py
   python test_uniform_sampling_with_variable_resources.py --nworkers 3

The number of concurrent evaluations of the objective function will be 4-1=3.

Note: This test contains multiple iterations to test different configurations.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_EXTRA: true

import sys
from multiprocessing import set_start_method

import numpy as np

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.sampling import uniform_random_sample_with_variable_resources as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import helloworld, six_hump_camel
from libensemble.sim_funcs.var_resources import multi_points_with_variable_resources as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["sim_dirs_make"] = True
    en_suffix = str(nworkers) + "_" + libE_specs.get("comms")
    libE_specs["ensemble_dir_path"] = "./ensemble_diff_nodes_w" + en_suffix

    if libE_specs["comms"] == "tcp":
        sys.exit("This test only runs with MPI or local -- aborting...")

    # Get paths for applications to run
    hello_world_app = helloworld.__file__
    six_hump_camel_app = six_hump_camel.__file__

    # Sim can run either helloworld or six_hump_camel
    exctr = MPIExecutor()
    exctr.register_app(full_path=hello_world_app, app_name="helloworld")
    exctr.register_app(full_path=six_hump_camel_app, app_name="six_hump_camel")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"app": "helloworld"},  # helloworld or six_hump_camel
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [
            ("priority", float),
            ("resource_sets", int),  # Set in gen func, resourced by alloc func.
            ("x", float, n),
            ("x_on_cube", float, n),
        ],
        "user": {
            "gen_batch_size": 5,
            "max_resource_sets": nworkers,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {
        "alloc_f": give_sim_work_first,
        "user": {
            "batch_mode": False,
            "give_all_with_same_priority": True,
            "num_active_gens": 1,
            "async_return": True,
        },
    }

    # This can improve scheduling when tasks may run across multiple nodes
    libE_specs["scheduler_opts"] = {"match_slots": False}

    exit_criteria = {"sim_max": 40, "wallclock_max": 300}

    if libE_specs["comms"] == "local":
        iterations = 4
    else:
        iterations = 2

    for prob_id in range(iterations):
        if prob_id == 0:
            sim_specs["user"]["app"] = "six_hump_camel"
        else:
            sim_specs["user"]["app"] = "helloworld"
            if prob_id == 1:
                libE_specs["ensemble_dir_path"] = "ensemble_hw_fork" + en_suffix
                set_start_method("fork", force=True)
            elif prob_id == 2:
                libE_specs["ensemble_dir_path"] = "ensemble_hw_spawn" + en_suffix
                set_start_method("spawn", force=True)
            else:
                libE_specs["ensemble_dir_path"] = "ensemble_hw_forkserver" + en_suffix
                set_start_method("forkserver", force=True)

        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

        if is_manager:
            assert flag == 0
            save_libE_output(H, persis_info, __file__, nworkers)
