"""
Runs libEnsemble with a non-persistent generator performing uniform random
sampling.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_uniform_sampling.py
   python test_uniform_sampling.py --nworkers 3
   python test_uniform_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import datetime
import os

import numpy as np

from libensemble.gen_funcs.sampling import uniform_random_sample

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.mock_sim import mock_sim
from libensemble.sim_funcs.six_hump_camel import six_hump_camel
from libensemble.tests.regression_tests.common import read_generated_file
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from libensemble.tools import add_unique_random_streams, parse_args

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["save_every_k_sims"] = 400
    libE_specs["save_every_k_gens"] = 300
    libE_specs["save_H_with_date"] = True
    libE_specs["H_file_prefix"] = "TESTING"

    sim_specs = {
        "sim_f": six_hump_camel,  # Function whose output is being minimized
        "in": ["x"],  # Keys to be given to sim_f
        "out": [("f", float)],  # Name of the outputs from sim_f
    }
    # end_sim_specs_rst_tag

    gen_specs = {
        "gen_f": uniform_random_sample,  # Function generating sim_f input
        "out": [("x", float, (2,))],  # Tell libE gen_f output, type, size
        "user": {
            "gen_batch_size": 500,  # Used by this specific gen_f
            "lb": np.array([-3, -2]),  # Used by this specific gen_f
            "ub": np.array([3, 2]),  # Used by this specific gen_f
        },
    }
    # end_gen_specs_rst_tag

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"gen_max": 501, "wallclock_max": 300}

    for run in range(2):
        if run == 1:
            # Test running a mock sim using previous history file
            sim_specs["sim_f"] = mock_sim
            hfile = read_generated_file("TESTING_*_after_gen_1000.npy")
            sim_specs["user"] = {"history_file": hfile}

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

        if is_manager:
            assert flag == 0

            tol = 0.1
            for m in minima:
                assert np.min(np.sum((H["x"] - m) ** 2, 1)) < tol

            npy_files = [i for i in os.listdir(os.path.dirname(__file__)) if i.endswith(".npy")]
            date = str(datetime.datetime.today()).split(" ")[0]
            assert any([i.startswith("TESTING_") for i in npy_files])
            assert any([date in i for i in npy_files])

            print("\nlibEnsemble found the 6 minima within a tolerance " + str(tol))
