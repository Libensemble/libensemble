"""
Runs libEnsemble 1D sampling test with worker profiling.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling_with_profile.py
   python test_1d_sampling_with_profile.py --nworkers 3 --comms local
   python test_1d_sampling_with_profile.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_EXTRA: true

import numpy as np
import os
import time

from libensemble.libE import libE
from libensemble.sim_funcs.one_d_func import one_d_example as sim_f
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
from libensemble.tools import parse_args, add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["profile"] = True

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "out": [("x", float, (1,))],
        "user": {
            "gen_batch_size": 500,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"gen_max": 501}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert len(H) >= 501
        print("\nlibEnsemble with random sampling has generated enough points")

        assert "manager.prof" in os.listdir(), "Expected manager profile not found after run"
        os.remove("manager.prof")

        prof_files = [f"worker_{i+1}.prof" for i in range(nworkers)]

        # Ensure profile writes complete before checking
        time.sleep(0.5)

        for file in prof_files:
            assert file in os.listdir(), "Expected profile {file} not found after run"
            with open(file, "r") as f:
                data = f.read().split()
                num_worker_funcs_profiled = sum(["worker" in i for i in data])
            assert num_worker_funcs_profiled >= 8, (
                "Insufficient number of " + "worker functions profiled: " + str(num_worker_funcs_profiled)
            )

            os.remove(file)
