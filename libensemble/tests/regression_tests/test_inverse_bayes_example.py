"""
Tests libEnsemble's inverse_bayes generator function

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_inverse_bayes_example.py
   python test_inverse_bayes_example.py --nworkers 3 --comms local
   python test_inverse_bayes_example.py --nworkers 3 --comms tcp

Debugging:
   mpiexec -np 4 xterm -e "python inverse_bayes_example.py"

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

import sys
import numpy as np

from libensemble.libE import libE
from libensemble.sim_funcs.inverse_bayes import likelihood_calculator as sim_f
from libensemble.gen_funcs.persistent_inverse_bayes import persistent_updater_after_likelihood as gen_f
from libensemble.alloc_funcs.inverse_bayes_allocf import only_persistent_gens_for_inverse_bayes as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    # Parse args for test code
    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("like", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": [],
        "out": [
            ("x", float, 2),
            ("batch", int),
            ("subbatch", int),
            ("prior", float),
            ("prop", float),
            ("weight", float),
        ],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "subbatch_size": 3,
            "num_subbatches": 2,
            "num_batches": 10,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Tell libEnsemble when to stop
    val = gen_specs["user"]["subbatch_size"] * gen_specs["user"]["num_subbatches"] * gen_specs["user"]["num_batches"]
    exit_criteria = {
        "sim_max": val,
        "wallclock_max": 300,
    }

    alloc_specs = {"out": [], "alloc_f": alloc_f}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert flag == 0
        # Change the last weights to correct values (H is a list on other cores and only array on manager)
        ind = 2 * gen_specs["user"]["subbatch_size"] * gen_specs["user"]["num_subbatches"]
        H[-ind:] = H["prior"][-ind:] + H["like"][-ind:] - H["prop"][-ind:]
        assert len(H) == 60, "Failed"
