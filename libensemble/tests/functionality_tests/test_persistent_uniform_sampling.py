"""
Tests libEnsemble with a simple persistent uniform sampling generator
function.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_sampling.py
   python test_persistent_uniform_sampling.py --nworkers 3 --comms local
   python test_persistent_uniform_sampling.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f1
from libensemble.gen_funcs.persistent_sampling import batched_history_matching as gen_f2
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    batch = 20
    num_batches = 10

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("grad", float, n)],
    }

    gen_specs = {
        "persis_in": ["x", "f", "grad", "sim_id"],
        "out": [("x", float, (n,))],
        "user": {
            "initial_batch_size": batch,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    exit_criteria = {"gen_max": num_batches * batch, "wallclock_max": 300}

    libE_specs["kill_canceled_sims"] = False

    for run in range(2):
        persis_info = add_unique_random_streams({}, nworkers + 1)
        for i in persis_info:
            persis_info[i]["get_grad"] = True

        if run == 0:
            gen_specs["gen_f"] = gen_f1
        elif run == 1:
            gen_specs["gen_f"] = gen_f2
            gen_specs["user"]["num_best_vals"] = 5

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            assert len(np.unique(H["gen_ended_time"])) == num_batches

            save_libE_output(H, persis_info, __file__, nworkers)
