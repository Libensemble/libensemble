"""
Runs libEnsemble 1D sampling test with worker profiling.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling_with_profile.py
   python test_1d_sampling_with_profile.py --nworkers 3
   python test_1d_sampling_with_profile.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import os
import time

import numpy as np

from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
from libensemble.libE import libE
from libensemble.sim_funcs.simple_sim import norm_eval as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["profile"] = True
    libE_specs["safe_mode"] = False
    libE_specs["kill_canceled_sims"] = False

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

    exit_criteria = {"sim_max": 501}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert len(H) >= 501
        print("\nlibEnsemble with random sampling has generated enough points")

        assert "manager.prof" in os.listdir(), "Expected manager profile not found after run"

        prof_files = [f"worker_{i + 1}.prof" for i in range(nworkers)]

        # Ensure profile writes complete before checking
        time.sleep(0.5)

        for file in prof_files:
            assert file in os.listdir(), "Expected profile {file} not found after run"
