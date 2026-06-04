"""
Runs libEnsemble with APOSMM (direct implementation) + DFO-LS on the chwirut
least-squares problem. All 214 residual calculations for a given point are
performed as a single simulation evaluation.

Uses GenSpecs(generator=APOSMM(...)) with the legacy chwirut_eval sim_f, as
the fvec-returning sim interface is not yet expressible in pure gest-api VOCS.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_dfols.py
   python test_persistent_aposmm_dfols.py --nworkers 3
   python test_persistent_aposmm_dfols.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as the generator runs on the manager thread.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "dfols"

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.sim_funcs.chwirut1 import EvaluateFunction, EvaluateJacobian
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    workflow = Ensemble(parse_args=True)

    # Declare the run parameters
    m = 214  # number of residual components
    n = 3  # number of variables

    lb = (-2 - np.pi / 10) * np.ones(n)
    ub = 2 * np.ones(n)

    vocs = VOCS(
        variables={**{f"x{i}": [lb[i], ub[i]] for i in range(n)}, **{f"x{i}_cube": [0, 1] for i in range(n)}},
        objectives={"f": "MINIMIZE"},
    )

    variables_mapping = {
        "x": [f"x{i}" for i in range(n)],
        "x_on_cube": [f"x{i}_cube" for i in range(n)],
        "f": ["f"],
        "fvec": ["fvec"],
    }

    aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=100,
        variables_mapping=variables_mapping,
        localopt_method="dfols",
        components=m,
        dfols_kwargs={
            "do_logging": False,
            "rhoend": 1e-5,
            "user_params": {
                "model.abs_tol": 1e-10,
                "model.rel_tol": 1e-4,
            },
        },
    )

    # Use legacy sim_f since chwirut_eval returns fvec as a numpy array field
    workflow.gen_specs = GenSpecs(
        generator=aposmm,
        vocs=vocs,
        batch_size=5,
        initial_batch_size=100,
    )
    workflow.sim_specs = SimSpecs(
        sim_f=sim_f,
        inputs=["x"],
        outputs=[("f", float), ("fvec", float, m)],
        user={},
    )
    # Tell libEnsemble when to stop (stop_val key must be in H)
    workflow.exit_criteria = ExitCriteria(sim_max=1000, wallclock_max=100, stop_val=("f", 3000))

    # Perform the run
    H, persis_info, flag = workflow.run()

    if workflow.is_manager:
        assert flag == 0
        assert np.min(H["f"][H["sim_ended"]]) <= 3000, "Didn't find a value below 3000"

        # Verify Jacobian at local minima (optional diagnostic)
        for i in np.where(H["local_min"])[0]:
            EvaluateFunction(H["x"][i])
            EvaluateJacobian(H["x"][i])
