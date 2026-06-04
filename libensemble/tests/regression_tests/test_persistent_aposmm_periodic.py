"""
Tests the 'periodic' domain use case for APOSMM (direct implementation) with both
NLopt and SciPy local optimization methods.

Uses the Ensemble/GenSpecs/SimSpecs dataclass interface with VOCS parameterization.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_periodic.py
   python test_persistent_aposmm_periodic.py --nworkers 3

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as the generator runs on the manager thread.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing

import numpy as np
from numpy import cos, sin

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = ["nlopt", "scipy"]

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs


def periodic_sim(x):
    """Periodic objective: sin(x0) * cos(x1), gest-api style (dict in, dict out)."""
    return {"f_val": sin(x["x0"]) * cos(x["x1"])}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    workflow = Ensemble(parse_args=True)

    lb = np.array([0.0, -np.pi / 2])
    ub = np.array([2 * np.pi, 3 * np.pi / 2])

    vocs = VOCS(
        variables={
            "x0": [lb[0], ub[0]],
            "x1": [lb[1], ub[1]],
            "x0_cube": [0, 1],
            "x1_cube": [0, 1],
        },
        objectives={"f_val": "MINIMIZE"},
    )

    variables_mapping = {
        "x": ["x0", "x1"],
        "x_on_cube": ["x0_cube", "x1_cube"],
        "f": ["f_val"],
    }

    # Known minima on unit cube for this periodic function
    periodic_minima = np.array([[0.25, 0.75], [0.75, 0.25]])
    tol = 2e-4
    exit_criteria = ExitCriteria(sim_max=1000)

    for run in range(2):
        if run == 0:
            # Run 0: NLopt LN_BOBYQA
            aposmm = APOSMM(
                vocs,
                max_active_runs=6,
                initial_sample_size=100,
                variables_mapping=variables_mapping,
                localopt_method="LN_BOBYQA",
                xtol_abs=1e-8,
                ftol_abs=1e-8,
                periodic=True,
            )
        else:
            # Run 1: SciPy COBYLA with scipy_kwargs
            aposmm = APOSMM(
                vocs,
                max_active_runs=6,
                initial_sample_size=100,
                variables_mapping=variables_mapping,
                localopt_method="scipy_COBYLA",
                opt_return_codes=[1],
                periodic=True,
                scipy_kwargs={"tol": 1e-8},
            )

        workflow.gen_specs = GenSpecs(
            generator=aposmm,
            vocs=vocs,
            batch_size=5,
            initial_batch_size=100,
        )
        workflow.sim_specs = SimSpecs(simulator=periodic_sim, vocs=vocs)
        workflow.exit_criteria = exit_criteria

        H, persis_info, flag = workflow.run()

        if workflow.is_manager:
            min_ids = np.where(H["local_min"])

            for x in H["x_on_cube"][min_ids]:
                print(x)
                print(np.linalg.norm(x - periodic_minima[0]))
                print(np.linalg.norm(x - periodic_minima[1]), flush=True)

            for x in H["x_on_cube"][min_ids]:
                assert (
                    np.linalg.norm(x - periodic_minima[0]) < tol or np.linalg.norm(x - periodic_minima[1]) < tol
                ), f"Run {run}: found minimum at {x} not near known periodic minima"
