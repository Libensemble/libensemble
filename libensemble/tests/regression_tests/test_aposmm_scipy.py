"""
Runs libEnsemble with APOSMM and SciPy local optimization routines.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_aposmm_scipy.py
   python test_aposmm_scipy.py --nworkers 3 --comms local
   python test_aposmm_scipy.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as the generator runs on the manager.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
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
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        nu=1e-8,
        mu=1e-8,
        dist_to_bound_multiple=0.01,
    )

    workflow.gen_specs = GenSpecs(
        generator=aposmm,
        vocs=vocs,
        initial_batch_size=100,
    )

    workflow.sim_specs = SimSpecs(simulator=six_hump_camel_func, vocs=vocs)
    workflow.exit_criteria = ExitCriteria(sim_max=1000)

    H, _, _ = workflow.run()

    if workflow.is_manager:
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
