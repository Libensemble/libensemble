"""
Runs libEnsemble with APOSMM (direct implementation) with the NLopt LD_MMA optimizer,
which uses gradient information from the simulator.

Demonstrates H0 preloading: pre-evaluated sample points are passed via the
``History`` parameter of APOSMM, setting ``initial_sample_size=0``.

Demonstrates early stopping via ``stop_after_k_minima``.

Uses the Ensemble/GenSpecs/SimSpecs dataclass interface with VOCS parameterization.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_with_grad.py
   python test_persistent_aposmm_with_grad.py --nworkers 3
   python test_persistent_aposmm_with_grad.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as the generator runs on the manager thread.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
from math import gamma, pi, sqrt

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from time import time

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func, six_hump_camel_grad
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima


def sim_with_grad(x):
    """Six-hump camel: return objective and gradient (gest-api style)."""
    x_arr = np.array([x["core"], x["edge"]])
    return {
        "energy": six_hump_camel_func(x_arr),
        "grad_core": six_hump_camel_grad(x_arr)[0],
        "grad_edge": six_hump_camel_grad(x_arr)[1],
    }


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    workflow = Ensemble(parse_args=True)

    if workflow.is_manager:
        start_time = time()

    n = 2
    lb = np.array([-3.0, -2.0])
    ub = np.array([3.0, 2.0])

    vocs = VOCS(
        variables={
            "core": [lb[0], ub[0]],
            "edge": [lb[1], ub[1]],
            "core_on_cube": [0, 1],
            "edge_on_cube": [0, 1],
        },
        objectives={"energy": "MINIMIZE"},
        observables=["grad_core", "grad_edge"],
    )

    variables_mapping = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube", "edge_on_cube"],
        "f": ["energy"],
        "grad": ["grad_core", "grad_edge"],
    }

    # Build H0: pre-evaluated sample points (known six-hump camel minima, rounded)
    # Two points with the same best function value test a corner case in APOSMM logic.
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
    H0["x"] = np.round(minima, 1)
    H0["x_on_cube"] = (H0["x"] - lb) / (ub - lb)
    H0["sim_id"] = range(sample_size)
    H0[["sim_started", "gen_informed", "sim_ended"]] = True
    for i in range(sample_size):
        H0["f"][i] = six_hump_camel_func(H0["x"][i])
        H0["grad"][i] = six_hump_camel_grad(H0["x"][i])

    aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=0,
        History=H0,
        variables_mapping=variables_mapping,
        localopt_method="LD_MMA",
        rk_const=0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
        stop_after_k_minima=15,
        xtol_rel=1e-6,
        ftol_rel=1e-6,
    )

    workflow.gen_specs = GenSpecs(
        generator=aposmm,
        vocs=vocs,
        batch_size=5,
        initial_batch_size=0,
    )
    workflow.sim_specs = SimSpecs(simulator=sim_with_grad, vocs=vocs)
    workflow.exit_criteria = ExitCriteria(sim_max=1000)

    # Perform the run
    H, persis_info, flag = workflow.run()

    if workflow.is_manager:
        assert len(H) < 1000, "Test should have stopped early due to 'stop_after_k_minima'"

        print("[Manager]:", H[np.where(H["local_min"])]["x"])
        print("[Manager]: Time taken =", time() - start_time, flush=True)

        tol = 1e-5
        for m in minima:
            print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
            assert np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol
