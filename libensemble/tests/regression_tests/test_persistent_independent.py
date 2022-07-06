"""
Tests the persistent_independent_optimize generator function.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_independent.py
   python test_persistent_independent.py --nworkers 3 --comms local

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 6
# TESTSUITE_OS_SKIP: OSX
# TESTSUITE_EXTRA: true


import sys
import numpy as np
import scipy.sparse as spp

from libensemble.libE import libE
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.gen_funcs.persistent_independent_optimize import independent_optimize as gen_f
from libensemble.alloc_funcs.start_persistent_consensus import start_consensus_persistent_gens as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.tools.consensus_subroutines import get_k_reach_chain_matrix

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()
    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")
    if nworkers < 5:
        sys.exit(
            "This tests requires at least 5 workers (6 MPI processes). You can \
                decrease the number of workers by modifying the number of gens and \
                communication graph @A in the calling script."
        )

    m = 16
    n = 32
    num_gens = 4
    eps = 1e-2

    # Even though we do not use consensus matrix, we still need to pass into alloc
    A = spp.diags([1, 2, 2, 1]) - get_k_reach_chain_matrix(num_gens, 1)

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x", "obj_component", "get_grad"],
        "out": [
            ("f_i", float),
            ("gradf_i", float, (n,)),
        ],
    }

    # lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
    gen_specs = {
        "gen_f": gen_f,
        "out": [
            ("x", float, (n,)),
            ("f_i", float),
            ("eval_pt", bool),  # eval point
            ("consensus_pt", bool),  # does not require a sim
            ("obj_component", int),  # which {f_i} to eval
            ("get_grad", bool),
            ("resource_sets", int),  # Just trying to cover in the alloc_f, not actually used
        ],
        "user": {
            "lb": np.array([-1.2, 1] * (n // 2)),
            "ub": np.array([-1.2, 1] * (n // 2)),
        },
    }

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "m": m,
            "num_gens": num_gens,
        },
    }

    persis_info = {}
    persis_info = add_unique_random_streams(persis_info, nworkers + 1)
    persis_info["gen_params"] = {"eps": eps}
    persis_info["sim_params"] = {"const": 1}
    persis_info["A"] = A

    assert n == 2 * m, "@n must be double of @m"

    # Perform the run
    libE_specs["safe_mode"] = False

    # i==0 is full run, i==1 is early termination
    for i in range(2):
        if i == 0:
            exit_criteria = {"wallclock_max": 600, "sim_max": 1000000}
            if is_manager:
                print("=== Testing full independent optimize ===", flush=True)
        else:
            exit_criteria = {"wallclock_max": 600, "sim_max": 10}
            if is_manager:
                print("=== Testing independent optimize w/ stoppage ===", flush=True)

        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            print("=== End algorithm ===", flush=True)

            # check we completed
            assert flag == 0

        if is_manager and i == 0:
            # compile sum of {f_i} and {x}, and check their values are bounded by O(eps)
            eval_H = H[H["eval_pt"]]

            gen_ids = np.unique(eval_H["gen_worker"])
            assert len(gen_ids) == num_gens, "Gen did not submit any function eval requests"

            F = 0
            fstar = 0

            for i, gen_id in enumerate(gen_ids):
                last_eval_idx = np.where(eval_H["gen_worker"] == gen_id)[0][-1]
                f_i = eval_H[last_eval_idx]["f_i"]
                F += f_i

            assert F - fstar < eps, "Error of {:.4e}, expected {:.4e} (assuming f*={:.4e})".format(
                F - fstar, eps, fstar
            )
