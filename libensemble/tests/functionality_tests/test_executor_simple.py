"""
Runs libEnsemble testing the executor functionality.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_executor_hworld.py
   python test_executor_hworld.py --nworkers 3
   python test_executor_hworld.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import numpy as np

import libensemble.sim_funcs.six_hump_camel as six_hump_camel
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.libE import libE

# Import libEnsemble items for this test
from libensemble.message_numbers import WORKER_DONE
from libensemble.sim_funcs.executor_hworld import executor_hworld as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_OMPI_SKIP: true

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    cores_per_task = 1
    cores_all_tasks = nworkers * cores_per_task

    sim_app2 = six_hump_camel.__file__

    exctr = MPIExecutor()
    exctr.register_app(full_path=sim_app2, app_name="six_hump_camel", calc_type="sim")  # Named app

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("cstat", int)],
        "user": {"cores": cores_per_task},
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

    # num sim_ended_count conditions in executor_hworld
    exit_criteria = {"sim_max": nworkers * 5}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        print("\nChecking expected task status against Workers ...\n")

        calc_status_list_in = np.asarray([WORKER_DONE] * 5)
        calc_status_list = np.repeat(calc_status_list_in, nworkers)

        # For debug
        print(f"Expecting: {calc_status_list}")
        print(f"Received:  {H['cstat']}\n")

        assert np.array_equal(H["cstat"], calc_status_list), "Error - unexpected calc status. Received: " + str(
            H["cstat"]
        )

        print("\n\n\nRun completed.")
