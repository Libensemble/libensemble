"""
Runs libEnsemble with APOSMM+POUNDERS on the chwirut least squares problem.
All 214 residual calculations for a given point are performed as a single
simulation evaluation.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_pounders.py
   python test_persistent_aposmm_pounders.py --nworkers 3
   python test_persistent_aposmm_pounders.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
import sys
from math import ceil, gamma, pi, sqrt

import numpy as np

import libensemble.gen_funcs

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f

libensemble.gen_funcs.rc.aposmm_optimizers = "petsc"
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.gen_funcs.sampling import lhs_sample
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output


def combine_component(x):
    return np.sum(np.power(x, 2))


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    # Declare the run parameters/functions
    m = 214
    n = 3
    budget = 10

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("fvec", float, m)],
        "user": {
            "combine_component_func": combine_component,
        },
    }

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    # lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
    lb = (-2 - np.pi / 10) * np.ones(n)
    ub = 2 * np.ones(n)

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "fvec"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "localopt_method": "pounders",
            "rk_const": 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            "grtol": 1e-6,
            "gatol": 1e-6,
            "dist_to_bound_multiple": 0.5,
            "lhs_divisions": 100,
            "components": m,
            "lb": lb,
            "ub": ub,
        },
    }

    alloc_specs = {"alloc_f": alloc_f, "user": {"batch_mode": True, "num_active_gens": 1}}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 500}

    sample_points = np.zeros((0, n))
    rand_stream = np.random.default_rng(0)
    for i in range(ceil(exit_criteria["sim_max"] / gen_specs["user"]["lhs_divisions"])):
        sample_points = np.append(sample_points, lhs_sample(n, gen_specs["user"]["lhs_divisions"], rand_stream), axis=0)

    gen_specs["user"]["sample_points"] = sample_points * (ub - lb) + lb

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert flag == 0
        assert len(H) >= budget

        save_libE_output(H, persis_info, __file__, nworkers)
        # # Calculating the Jacobian at the best point (though this information was not used by pounders)
        # from libensemble.sim_funcs.chwirut1 import EvaluateJacobian
        # J = EvaluateJacobian(H['x'][np.argmin(H['f'])])
        # assert np.linalg.norm(J) < 2000
