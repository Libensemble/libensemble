"""
Runs libEnsemble with uniform random sampling and writes results into sim dirs.
This tests when an exception occurs in sim_dir capabilities.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_sim_dirs_with_exception.py
   python test_sim_dirs_with_exception.py --nworkers 3
   python test_sim_dirs_with_exception.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import os

import numpy as np

from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.libE import libE
from libensemble.manager import LoggedException
from libensemble.tests.regression_tests.support import write_sim_func as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

nworkers, is_manager, libE_specs, _ = parse_args()

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    e_ensemble = "./ensemble_ex_w" + str(nworkers) + "_" + libE_specs.get("comms")

    if not os.path.isdir(e_ensemble):
        os.makedirs(os.path.join(e_ensemble, "sim0"), exist_ok=True)

    libE_specs["sim_dirs_make"] = True
    libE_specs["ensemble_dir_path"] = e_ensemble
    libE_specs["abort_on_exception"] = False

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "out": [("x", float, (1,))],
        "user": {
            "gen_batch_size": 20,
            "lb": np.array([-3]),
            "ub": np.array([3]),
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 21}

    return_flag = 1
    try:
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)
    except LoggedException as e:
        print(f"Caught deliberate exception: {e}")
        return_flag = 0

    if is_manager:
        assert return_flag == 0
