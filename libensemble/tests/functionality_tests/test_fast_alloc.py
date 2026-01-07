"""
Tests various capabilities of the libEnsemble fast_alloc alloc_f

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_fast_alloc.py
   python test_fast_alloc.py --nworkers 3

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import gc
import sys

import numpy as np

from libensemble.alloc_funcs.fast_alloc import give_sim_work_first as alloc_f
from libensemble.alloc_funcs.only_one_gen_alloc import ensure_one_active_gen as alloc_f2
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.branin.branin_obj import call_branin as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    num_pts = 30 * (nworkers)

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("large", float, 1000000)],
        "user": {},
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, (2,))],
        "user": {
            "gen_batch_size": num_pts,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 2 * num_pts, "wallclock_max": 300}

    if libE_specs["comms"] == "tcp":
        # Can't use the same interface for manager and worker if we want
        # repeated calls to libE -- the manager sets up a different server
        # each time, and the worker will not know what port to connect to.
        sys.exit("Cannot run with tcp when repeated calls to libE -- aborting...")

    for time in np.append([0], np.logspace(-5, -1, 2)):
        if is_manager:
            print("Starting for time: ", time, flush=True)
        if time == 0:
            alloc_specs = {"alloc_f": alloc_f2}
        else:
            alloc_specs = {"alloc_f": alloc_f, "user": {"num_active_gens": 1}}

        for rep in range(1):
            sim_specs["user"]["uniform_random_pause_ub"] = time

            if time == 0:
                sim_specs["user"].pop("uniform_random_pause_ub")
                gen_specs["user"]["gen_batch_size"] = num_pts // 2

            persis_info["next_to_give"] = 0
            persis_info["total_gen_calls"] = 1

            H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

            if is_manager:
                assert flag == 0
                assert len(H) == 2 * num_pts

            del H
            gc.collect()  # If doing multiple libE calls, users might need to clean up their memory space.
