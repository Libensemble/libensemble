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
# TESTSUITE_NPROCS: 4

import datetime
import os

import numpy as np
from gest_api.vocs import VOCS

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.gen_funcs.sampling import uniform_random_sample

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.mock_sim import mock_sim
from libensemble.sim_funcs.six_hump_camel import six_hump_camel
from libensemble.tests.regression_tests.common import read_generated_file
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from libensemble.tools import parse_args

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

    vocs = VOCS(variables={"x0": [-3, 3], "x1": [-2, 2]}, objectives={"f": "EXPLORE"})

    gen_specs = {
        "gen_f": uniform_random_sample,  # Function generating sim_f input
        "out": [("x", float, (2,))],  # Tell libE gen_f output, type, size
        "batch_size": 500,
        "vocs": vocs,
    }
    # end_gen_specs_rst_tag

    exit_criteria = {"gen_max": 501, "wallclock_max": 300}

    alloc_specs = {
        "alloc_f": give_sim_work_first,
    }

    for run in range(2):
        if run == 1:
            # Test running a mock sim using previous history file
            sim_specs["sim_f"] = mock_sim
            hfile = read_generated_file("TESTING_*_after_gen_1000.npy")
            sim_specs["user"] = {"history_file": hfile}

        # Perform the run
        H, _, flag = libE(
            sim_specs,
            gen_specs,
            exit_criteria,
            alloc_specs=alloc_specs,
            libE_specs=libE_specs,
        )

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
