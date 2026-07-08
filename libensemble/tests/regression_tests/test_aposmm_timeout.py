"""
Test the APOSMM generator's ability to properly exit when a timeout has occurred.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_aposmm_timeout.py
   python test_aposmm_timeout.py --nworkers 3 --comms local
   python test_aposmm_timeout.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as the generator runs on the manager.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs


def periodic_func(x):
    """
    Periodic test function (gest-api version of periodic_func.func_wrapper).
    """
    from numpy import cos, sin

    return {"f": sin(x["x0"]) * cos(x["x1"])}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    workflow = Ensemble(parse_args=True)

    vocs = VOCS(
        variables={
            "x0": [0, 2 * np.pi],
            "x1": [-np.pi / 2, 3 * np.pi / 2],
            "x0_on_cube": [0, 1],
            "x1_on_cube": [0, 1],
        },
        objectives={"f": "MINIMIZE"},
    )

    aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=100,
        variables_mapping={
            "x": ["x0", "x1"],
            "x_on_cube": ["x0_on_cube", "x1_on_cube"],
            "f": ["f"],
        },
        localopt_method="LN_BOBYQA",
        xtol_abs=1e-8,
        ftol_abs=1e-8,
        periodic=True,
        print=True,
    )

    workflow.gen_specs = GenSpecs(
        generator=aposmm,
        vocs=vocs,
        initial_batch_size=100,
    )
    workflow.sim_specs = SimSpecs(simulator=periodic_func, vocs=vocs)

    # Setting a very high sim_max and a short wallclock_max so timeout will occur
    workflow.exit_criteria = ExitCriteria(sim_max=50000, wallclock_max=5)

    H, _, flag = workflow.run()

    if workflow.is_manager:
        assert flag == 2, "Test should have timed out"
        assert np.any(H["local_min"]), "Expected at least one local minimum to be found"
