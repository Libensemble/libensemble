"""
Runs libEnsemble with APOSMM and SciPy local optimization routines.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_scipy.py
   python test_persistent_aposmm_scipy.py --nworkers 3
   python test_persistent_aposmm_scipy.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
import sys

import numpy as np

import libensemble.gen_funcs

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
from time import time

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

    if is_manager:
        start_time = time()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "sample_points": np.round(minima, 1),
            "localopt_method": "scipy_Nelder-Mead",
            "opt_return_codes": [0],
            "nu": 1e-8,
            "mu": 1e-8,
            "dist_to_bound_multiple": 0.01,
            "max_active_runs": 6,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    exit_criteria = {"sim_max": 1000}

    for run in range(2):
        persis_info = add_unique_random_streams({}, nworkers + 1)

        if run == 1:
            gen_specs["user"]["localopt_method"] = "scipy_BFGS"
            gen_specs["user"]["opt_return_codes"] = [0]
            gen_specs["persis_in"].append("grad")
            sim_specs["out"] = [("f", float), ("grad", float, n)]

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            print("[Manager]:", H[np.where(H["local_min"])]["x"])
            print("[Manager]: Time taken =", time() - start_time, flush=True)

            tol = 1e-3
            min_found = 0
            for m in minima:
                # The minima are known on this test problem.
                # We use their values to test APOSMM has identified all minima
                print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
                if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
                    min_found += 1
            assert min_found >= 2, f"Found {min_found} minima"

            save_libE_output(H, persis_info, __file__, nworkers)

    # Now let's run on the same problem with a really large n (but we won't test
    # convergence to all local min). Note that sim_f uses only entries x[0:2]
    n = 400
    persis_info = add_unique_random_streams({}, nworkers + 1)
    gen_specs["out"][0:2] = [("x", float, n), ("x_on_cube", float, n)]
    gen_specs["user"]["lb"] = np.zeros(n)
    gen_specs["user"]["ub"] = np.ones(n)
    gen_specs["user"]["lb"][:2] = [-3, -2]
    gen_specs["user"]["ub"][:2] = [3, 2]
    gen_specs["user"]["rk_const"] = 4.90247
    gen_specs["user"].pop("sample_points")
    gen_specs["user"]["localopt_method"] = "scipy_Nelder-Mead"
    sim_specs["out"] = [("f", float)]
    gen_specs["persis_in"].remove("grad")

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert np.sum(H["sim_ended"]) >= exit_criteria["sim_max"], "Run didn't finish"
