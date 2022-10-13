"""
Runs libEnsemble with a non-persistent generator performing uniform random
sampling.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_uniform_sampling.py
   python test_uniform_sampling.py --nworkers 3 --comms local
   python test_uniform_sampling.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["save_every_k_sims"] = 400
    libE_specs["save_every_k_gens"] = 300

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

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert flag == 0

        tol = 0.1
        for m in minima:
            assert np.min(np.sum((H["x"] - m) ** 2, 1)) < tol

        print("\nlibEnsemble found the 6 minima within a tolerance " + str(tol))
        save_libE_output(H, persis_info, __file__, nworkers)
