"""
Runs libEnsemble with APOSMM with the NLopt local optimizer.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_aposmm_nlopt.py
   python test_aposmm_nlopt.py --nworkers 3 --comms local
   python test_aposmm_nlopt.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as the generator runs on the manager.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4

from math import gamma, pi, sqrt

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from time import time

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x["x0"]
    x2 = x["x1"]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return {"f": term1 + term2 + term3}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    workflow = Ensemble(parse_args=True)

    if workflow.is_manager:
        start_time = time()

    n = 2

    vocs = VOCS(
        variables={
            "x0": [-3, 3],
            "x1": [-2, 2],
            "x0_on_cube": [0, 1],
            "x1_on_cube": [0, 1],
        },
        objectives={"f": "MINIMIZE"},
    )

    aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        variables_mapping={
            "x": ["x0", "x1"],
            "x_on_cube": ["x0_on_cube", "x1_on_cube"],
            "f": ["f"],
        },
        initial_sample_size=100,
        sample_points=np.round(minima, 1),
        localopt_method="LN_BOBYQA",
        rk_const=0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
        xtol_abs=1e-6,
        ftol_abs=1e-6,
        dist_to_bound_multiple=0.5,
    )

    workflow.gen_specs = GenSpecs(
        generator=aposmm,
        vocs=vocs,
        initial_batch_size=100,
    )

    workflow.sim_specs = SimSpecs(simulator=six_hump_camel_func, vocs=vocs)
    workflow.exit_criteria = ExitCriteria(sim_max=3000)

    # Perform the run
    H, _, _ = workflow.run()

    if workflow.is_manager:
        print("[Manager]:", H[np.where(H["local_min"])]["x"])
        print("[Manager]: Time taken =", time() - start_time, flush=True)

        tol = 1e-5
        for m in minima:
            # The minima are known on this test problem.
            # We use their values to test APOSMM has identified all minima
            print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
            assert np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol
