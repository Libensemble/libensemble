"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_sampling.py
   python test_1d_sampling.py --nworkers 3
   python test_1d_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 4

import numpy as np

from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.simple_sim import norm_eval as sim_f
from libensemble.tests.regression_tests.common import mpi_comm_excl
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["mpi_comm"], mpi_comm_null = mpi_comm_excl()

    if libE_specs["mpi_comm"] == mpi_comm_null:
        is_excluded = True
        is_manager = False
    else:
        is_manager = libE_specs["mpi_comm"].Get_rank() == 0
        is_excluded = False

    libE_specs["save_every_k_gens"] = 300
    libE_specs["safe_mode"] = False
    libE_specs["disable_log_files"] = True

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

    persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

    exit_criteria = {"gen_max": 501}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert len(H) >= 501
        assert flag == 0
        print("\nlibEnsemble with random sampling has generated enough points")
        save_libE_output(H, persis_info, __file__, nworkers)

    elif is_excluded:
        assert flag == 3
