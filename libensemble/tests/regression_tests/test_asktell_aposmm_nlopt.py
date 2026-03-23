"""
Runs libEnsemble with APOSMM with the NLopt local optimizer.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_nlopt.py
   python test_persistent_aposmm_nlopt.py --nworkers 3 --comms local
   python test_persistent_aposmm_nlopt.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 3

from math import gamma, pi, sqrt

import numpy as np

import libensemble.gen_funcs
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.sim_funcs import six_hump_camel

# Import libEnsemble items for this test

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from time import time

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_classes import APOSMM
from libensemble.manager import LoggedException
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, SimSpecs
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x["core"]
    x2 = x["edge"]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    return {"energy": term1 + term2 + term3}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    for run in range(3):

        workflow = Ensemble(parse_args=True)

        if workflow.is_manager:
            start_time = time()

        n = 2
        workflow.alloc_specs = AllocSpecs(alloc_f=alloc_f)

        workflow.libE_specs.gen_on_worker = False

        vocs = VOCS(
            variables={"core": [-3, 3], "edge": [-2, 2], "core_on_cube": [-3, 3], "edge_on_cube": [-2, 2]},
            objectives={"energy": "MINIMIZE"},
        )

        aposmm = APOSMM(
            vocs,
            max_active_runs=workflow.nworkers,  # should this match nworkers always? practically?
            variables_mapping={"x": ["core", "edge"], "x_on_cube": ["core_on_cube", "edge_on_cube"], "f": ["energy"]},
            initial_sample_size=100,
            sample_points=minima,
            localopt_method="LN_BOBYQA",
            rk_const=0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
            xtol_abs=1e-6,
            ftol_abs=1e-6,
        )

        workflow.gen_specs = GenSpecs(
            generator=aposmm,
            vocs=vocs,
            batch_size=5,
            initial_batch_size=10,
        )

        if run == 0:
            workflow.sim_specs = SimSpecs(simulator=six_hump_camel_func, vocs=vocs)
            workflow.exit_criteria = ExitCriteria(sim_max=2000)
        elif run == 1:
            workflow.persis_info["num_gens_started"] = 0
            sim_app2 = six_hump_camel.__file__
            exctr = MPIExecutor()
            exctr.register_app(full_path=sim_app2, app_name="six_hump_camel", calc_type="sim")  # Named app
            workflow.sim_specs = SimSpecs(simulator=six_hump_camel_func, vocs=vocs)
            workflow.exit_criteria = ExitCriteria(sim_max=200)
        elif run == 2:
            workflow.persis_info["num_gens_started"] = 0
            workflow.sim_specs = SimSpecs(
                sim_f=six_hump_camel_func, vocs=vocs
            )  # wrong parameter, but check we get error message
            workflow.exit_criteria = ExitCriteria(sim_max=200)
            workflow.libE_specs.abort_on_exception = False

        workflow.add_random_streams()

        try:
            H, _, _ = workflow.run()
        except Exception as e:
            if run == 2:
                assert isinstance(e, LoggedException)
                aposmm.finalize()
                print("Passed", flush=True)
            else:
                raise e

        # Perform the run
        if workflow.is_manager and run == 0:
            print("[Manager]:", H[np.where(H["local_min"])]["x"])
            print("[Manager]: Time taken =", time() - start_time, flush=True)

            tol = 1e-5
            for m in minima:
                # The minima are known on this test problem.
                # We use their values to test APOSMM has identified all minima
                print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
                assert np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol
