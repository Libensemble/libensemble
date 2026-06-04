"""
Runs libEnsemble with APOSMM (direct implementation) + POUNDERS on the chwirut
least-squares problem. All 214 residual calculations for a given point are
performed as a single simulation evaluation.

Uses GenSpecs(generator=APOSMM(...)) with the legacy chwirut_eval sim_f, as
the fvec-returning sim interface is not yet expressible in pure gest-api VOCS.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_pounders.py
   python test_persistent_aposmm_pounders.py --nworkers 3
   python test_persistent_aposmm_pounders.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3, as the generator runs on the manager thread.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
from math import ceil, gamma, pi, sqrt

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "petsc"

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.gen_funcs.sampling import lhs_sample
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs


def combine_component(x):
    return np.sum(np.power(x, 2))


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    workflow = Ensemble(parse_args=True)

    # Declare the run parameters
    m = 214  # number of residual components
    n = 3  # number of variables
    budget = 10

    # lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
    lb = (-2 - np.pi / 10) * np.ones(n)
    ub = 2 * np.ones(n)

    # VOCS for the n-dimensional chwirut problem
    vocs = VOCS(
        variables={f"x{i}": [lb[i], ub[i]] for i in range(n)},
        objectives={"f": "MINIMIZE"},
    )
    vocs_cube_vars = {f"x{i}_cube": [0, 1] for i in range(n)}
    vocs = VOCS(
        variables={**{f"x{i}": [lb[i], ub[i]] for i in range(n)}, **vocs_cube_vars},
        objectives={"f": "MINIMIZE"},
    )

    variables_mapping = {
        "x": [f"x{i}" for i in range(n)],
        "x_on_cube": [f"x{i}_cube" for i in range(n)],
        "f": ["f"],
        "fvec": ["fvec"],
    }

    # Build LHS sample points
    exit_sim_max = 500
    lhs_divisions = 100
    sample_points = np.zeros((0, n))
    rand_stream = np.random.default_rng(0)
    for i in range(ceil(exit_sim_max / lhs_divisions)):
        sample_points = np.append(sample_points, lhs_sample(n, lhs_divisions, rand_stream), axis=0)
    sample_points = sample_points * (ub - lb) + lb

    aposmm = APOSMM(
        vocs,
        max_active_runs=6,
        initial_sample_size=100,
        variables_mapping=variables_mapping,
        localopt_method="pounders",
        rk_const=0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
        grtol=1e-6,
        gatol=1e-6,
        dist_to_bound_multiple=0.5,
        lhs_divisions=lhs_divisions,
        components=m,
        sample_points=sample_points,
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
        user={"combine_component_func": combine_component},
    )
    workflow.exit_criteria = ExitCriteria(sim_max=exit_sim_max)

    # Perform the run
    H, persis_info, flag = workflow.run()

    if workflow.is_manager:
        assert flag == 0
        assert len(H) >= budget
