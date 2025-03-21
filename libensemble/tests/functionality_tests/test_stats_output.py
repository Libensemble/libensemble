"""
Runs libEnsemble doing uniform sampling with multiple tasks per simulation and
varying amount of resources, and then checks structure of the libE_stats.txt file.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_stats_output.py
   python test_stats_output.py --nworkers 3

The number of concurrent evaluations of the objective function will be 4-1=3.

Note: This test contains multiple iterations to test different libE_stats outputs.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import sys
import warnings

import numpy as np

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.sampling import uniform_random_sample_with_variable_resources as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs import helloworld, six_hump_camel
from libensemble.sim_funcs.var_resources import multi_points_with_variable_resources as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

warnings.filterwarnings("ignore", category=DeprecationWarning)
from check_libE_stats import check_libE_stats

# from libensemble.gen_funcs.sampling import uniform_random_sample_with_var_priorities_and_resources as gen_f


if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["sim_dirs_make"] = True

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

    iterations = 2

    # Note that libE_stats.txt output will be appended across libE calls.
    for prob_id in range(iterations):
        sim_specs["user"]["app"] = "six_hump_camel"

        libE_specs["ensemble_dir_path"] = (
            "./ensemble_test_stats" + str(nworkers) + "_" + libE_specs.get("comms") + "_" + str(prob_id)
        )

        if prob_id == 0:
            libE_specs["stats_fmt"] = {"task_timing": True}  # This adds total time for each task.
            check_task_datetime = False

        if prob_id == 1:
            # task_datetime: Include task_timing and start/end times for each task
            libE_specs["stats_fmt"] = {"task_datetime": True, "show_resource_sets": True}
            check_task_datetime = True

        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, persis_info, flag = libE(
            sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
        )

        if is_manager:
            assert flag == 0
            check_libE_stats(task_datetime=check_task_datetime)

            # save_libE_output(H, persis_info, __file__, nworkers)
