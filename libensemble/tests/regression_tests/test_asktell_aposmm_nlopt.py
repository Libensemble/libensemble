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

import sys
from math import gamma, pi, sqrt

import numpy as np

import libensemble.gen_funcs
from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.sim_funcs import six_hump_camel
from libensemble.sim_funcs.executor_hworld import executor_hworld as sim_f_exec

# Import libEnsemble items for this test
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from time import time

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    for run in range(2):

        workflow = Ensemble(parse_args=True)

        if workflow.is_manager:
            start_time = time()

        if workflow.nworkers < 2:
            sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

        n = 2

        vocs = VOCS(
            variables={"core": [-3, 3], "edge": [-2, 2], "core_on_cube": [-3, 3], "edge_on_cube": [-2, 2]},
            objectives={"energy": "MINIMIZE"},
        )

        workflow.libE_specs.gen_on_manager = True

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

        # SH TODO - dont want this stuff duplicated - pass with vocs instead
        workflow.gen_specs = GenSpecs(
            persis_in=["x", "x_on_cube", "sim_id", "local_min", "local_pt", "f"],
            generator=aposmm,
            batch_size=5,
            initial_batch_size=10,
            user={"initial_sample_size": 100},
        )

        if run == 0:
            workflow.sim_specs = SimSpecs(sim_f=sim_f, inputs=["x"], outputs=[("f", float)])
            workflow.exit_criteria = ExitCriteria(sim_max=2000)
        elif run == 1:
            workflow.persis_info["num_gens_started"] = 0
            sim_app2 = six_hump_camel.__file__
            exctr = MPIExecutor()
            exctr.register_app(full_path=sim_app2, app_name="six_hump_camel", calc_type="sim")  # Named app
            workflow.sim_specs = SimSpecs(
                sim_f=sim_f_exec, inputs=["x"], outputs=[("f", float), ("cstat", int)], user={"cores": 1}
            )
            workflow.exit_criteria = ExitCriteria(sim_max=200)

        workflow.add_random_streams()

        H, _, _ = workflow.run()

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
