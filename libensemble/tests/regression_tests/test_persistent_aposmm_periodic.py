"""
Tests the 'periodic' domain use case for APOSMM with both NLopt and SciPy
local optimization methods.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_periodic.py
   python test_persistent_aposmm_periodic.py --nworkers 3

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
import sys

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = ["nlopt", "scipy"]
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.periodic_func import func_wrapper as sim_f
from libensemble.tools import add_unique_random_streams, parse_args

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

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
            "localopt_method": "LN_BOBYQA",
            "xtol_abs": 1e-8,
            "ftol_abs": 1e-8,
            "lb": np.array([0, -np.pi / 2]),
            "ub": np.array([2 * np.pi, 3 * np.pi / 2]),
            "periodic": True,
            "print": True,
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    exit_criteria = {"sim_max": 1000}

    for run in range(2):
        if run == 1:
            gen_specs["user"]["localopt_method"] = "scipy_COBYLA"
            gen_specs["user"]["opt_return_codes"] = [1]
            gen_specs["user"].pop("xtol_abs")
            gen_specs["user"].pop("ftol_abs")
            gen_specs["user"]["scipy_kwargs"] = {"tol": 1e-8}

        persis_info = add_unique_random_streams({}, nworkers + 1)
        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            assert persis_info[1].get("run_order"), "Run_order should have been given back"
            min_ids = np.where(H["local_min"])

            # The minima are known on this test problem. If the above [lb, ub] domain is
            # shifted/scaled to [0,1]^n, they all have value [0.25, 0.75] or [0.75, 0.25]
            minima = np.array([[0.25, 0.75], [0.75, 0.25]])
            tol = 2e-4

            for x in H["x_on_cube"][min_ids]:
                print(x)
                print(np.linalg.norm(x - minima[0]))
                print(np.linalg.norm(x - minima[1]), flush=True)

            for x in H["x_on_cube"][min_ids]:
                assert np.linalg.norm(x - minima[0]) < tol or np.linalg.norm(x - minima[1]) < tol
