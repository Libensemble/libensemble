"""
Runs libEnsemble testing the executor functionality.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_executor_hworld.py
   python test_executor_hworld.py --nworkers 3
   python test_executor_hworld.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import multiprocessing
import os

import numpy as np

import libensemble.sim_funcs.six_hump_camel as six_hump_camel
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.executor_hworld import executor_hworld as sim_f
from libensemble.tests.regression_tests.common import build_simfunc
from libensemble.tools import add_unique_random_streams, parse_args

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 3 4
# TESTSUITE_OMPI_SKIP: true
# TESTSUITE_OS_SKIP: OSX WIN
# TESTSUITE_EXTRA: true

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["disable_resource_manager"] = True

    cores_per_task = 1
    logical_cores = multiprocessing.cpu_count()
    cores_all_tasks = nworkers * cores_per_task

    if cores_all_tasks > logical_cores:
        disable_resource_manager = True
        mess_resources = "Oversubscribing - Resource manager disabled"
    elif libE_specs.get("comms", False) == "tcp":
        disable_resource_manager = True
        mess_resources = "TCP comms does not support resource management. Resource manager disabled"
    else:
        disable_resource_manager = False
        mess_resources = "Resource manager enabled"

    if is_manager:
        print(f"\nCores req: {cores_all_tasks} Cores avail: {logical_cores}\n  {mess_resources}\n")

    sim_app = "./my_simtask.x"
    if not os.path.isfile(sim_app):
        build_simfunc()
    sim_app2 = six_hump_camel.__file__

    exctr = MPIExecutor()

    exctr.register_app(full_path=sim_app, calc_type="sim")  # Default 'sim' app - backward compatible
    exctr.register_app(full_path=sim_app2, app_name="six_hump_camel")  # Named app

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("cstat", int)],
        "user": {
            "cores": cores_per_task,
            "elapsed_timeout": True,
        },
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, (2,))],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "gen_batch_size": nworkers,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"wallclock_max": 10}

    # TCP does not support multiple libE calls
    if libE_specs["comms"] == "tcp":
        iterations = 1
    else:
        iterations = 2

    for i in range(iterations):
        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

        if is_manager:
            print("\nChecking expected task status against Workers ...\n")

            calc_status_list_in = np.asarray([0])
            calc_status_list = np.repeat(calc_status_list_in, nworkers)

            # For debug
            print(f"Expecting: {calc_status_list}")
            print(f"Received:  {H['cstat']}\n")

            assert np.array_equal(H["cstat"], calc_status_list), "Error - unexpected calc status. Received: " + str(
                H["cstat"]
            )

            print("\n\n\nRun completed.")
