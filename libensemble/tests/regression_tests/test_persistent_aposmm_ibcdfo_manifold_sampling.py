"""
Runs libEnsemble with APOSMM (direct implementation) + IBCDFO manifold sampling
on the synthetic beamline problem using a piecewise-maximum h-function.

Uses GenSpecs(generator=APOSMM(...)) with the legacy sim_f interface.

Execute via one of the following commands:
   mpiexec -np 3 python test_persistent_aposmm_ibcdfo_manifold_sampling.py
   python test_persistent_aposmm_ibcdfo_manifold_sampling.py --nworkers 2
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 3

import multiprocessing
import sys

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "ibcdfo_manifold_sampling"

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs

try:
    import ibcdfo  # noqa: F401
except ModuleNotFoundError:
    sys.exit("Please 'pip install ibcdfo'")

try:
    from minqsw import minqsw  # noqa: F401
except ModuleNotFoundError:
    sys.exit("Ensure https://github.com/POptUS/minq has been cloned and that minq/py/minq5/ is on the PYTHONPATH")


def synthetic_beamline_mapping(H, _, sim_specs):
    x = H["x"][0]
    assert len(x) == 4, "Assuming 4 inputs to this function"
    y = np.zeros(3)
    y[0] = x[0] ** 2 + 1.0
    y[1] = x[1] ** 2 + 2.0
    y[2] = x[2] * x[3] + 0.5

    Out = np.zeros(1, dtype=sim_specs["out"])
    Out["fvec"] = y
    Out["f"] = np.max(y)
    return Out


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    workflow = Ensemble(parse_args=True)

    assert workflow.nworkers == 2, "This test is just for two workers"

    m, n = 3, 4
    lb = -1 * np.ones(n)
    ub = np.ones(n)

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
        max_active_runs=1,
        initial_sample_size=1,
        variables_mapping=variables_mapping,
        localopt_method="ibcdfo_manifold_sampling",
        run_max_eval=100 * (n + 1),
        components=m,
        stop_after_k_runs=1,
        sample_points=np.atleast_2d(0.1 * (np.arange(n) + 1)),
        hfun=ibcdfo.manifold_sampling.h_pw_maximum,
    )

    workflow.gen_specs = GenSpecs(
        generator=aposmm,
        vocs=vocs,
        batch_size=1,
        initial_batch_size=1,
    )
    workflow.sim_specs = SimSpecs(
        sim_f=synthetic_beamline_mapping,
        inputs=["x"],
        outputs=[("f", float), ("fvec", float, m)],
    )
    workflow.exit_criteria = ExitCriteria(sim_max=500)

    H, persis_info, flag = workflow.run()

    if workflow.is_manager:
        assert np.min(H["f"][H["f"] > 0]) == 2.0, "The best is 2"
        assert flag == 0
