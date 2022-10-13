"""
Runs libEnsemble with APOSMM with an NLopt local optimizer that uses gradient
information from the sim_f

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_with_grad.py
   python test_persistent_aposmm_with_grad.py --nworkers 3 --comms local
   python test_persistent_aposmm_with_grad.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import sys
import multiprocessing
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from math import gamma, pi, sqrt
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f, six_hump_camel_func, six_hump_camel_grad

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from time import time

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
        "out": [("f", float), ("grad", float, n)],
    }

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_in = ["x", "f", "grad", "local_pt", "sim_id", "sim_ended", "x_on_cube", "local_min"]

    gen_specs = {
        "gen_f": gen_f,
        "in": gen_in,
        "persis_in": gen_in,
        "out": gen_out,
        "user": {
            "initial_sample_size": 0,  # Don't need to do evaluations because the sampling already done below
            "localopt_method": "LD_MMA",
            "rk_const": 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            "stop_after_this_many_minima": 25,
            "xtol_rel": 1e-6,
            "ftol_rel": 1e-6,
            "max_active_runs": 6,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 1000}

    # Load in "already completed" set of 'x','f','grad' values to give to libE/persistent_aposmm
    sample_size = len(minima)

    H0_dtype = [
        ("x", float, n),
        ("grad", float, n),
        ("sim_id", int),
        ("x_on_cube", float, n),
        ("sim_ended", bool),
        ("f", float),
        ("gen_informed", bool),
        ("sim_started", bool),
    ]
    H0 = np.zeros(sample_size, dtype=H0_dtype)

    # Two points in the following sample have the same best function value, which
    # tests the corner case for some APOSMM logic
    H0["x"] = np.round(minima, 1)
    H0["x_on_cube"] = (H0["x"] - gen_specs["user"]["lb"]) / (gen_specs["user"]["ub"] - gen_specs["user"]["lb"])
    H0["sim_id"] = range(sample_size)
    H0[["sim_started", "gen_informed", "sim_ended"]] = True

    for i in range(sample_size):
        H0["f"][i] = six_hump_camel_func(H0["x"][i])
        H0["grad"][i] = six_hump_camel_grad(H0["x"][i])

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs, H0=H0)

    if is_manager:
        print("[Manager]:", H[np.where(H["local_min"])]["x"])
        print("[Manager]: Time taken =", time() - start_time, flush=True)

        tol = 1e-5
        for m in minima:
            # The minima are known on this test problem.
            # We use their values to test APOSMM has identified all minima
            print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
            assert np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol

        assert len(H) < exit_criteria["sim_max"], "Test should have stopped early"

        save_libE_output(H, persis_info, __file__, nworkers)
