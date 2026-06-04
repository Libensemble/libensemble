"""
Runs libEnsemble with APOSMM (direct implementation) and SciPy local optimization routines.

Two runs are performed:
  Run 0: scipy_Nelder-Mead (derivative-free)
  Run 1: scipy_BFGS (gradient-based)

Additionally, a high-dimensional (n=400) run is performed to test scaling.

Uses the Ensemble/GenSpecs/SimSpecs dataclass interface with VOCS parameterization.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 5 python test_persistent_aposmm_scipy.py
   python test_persistent_aposmm_scipy.py --nworkers 4 --comms local
   python test_persistent_aposmm_scipy.py --nworkers 4 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 4, as the generator runs on the manager thread.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4

import multiprocessing

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"
from time import time

from gest_api.vocs import VOCS

from libensemble import Ensemble
from libensemble.gen_classes import APOSMM
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func, six_hump_camel_grad
from libensemble.specs import ExitCriteria, GenSpecs, SimSpecs
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima


def sim_func_f_only(x):
    """Six-hump camel: return objective only (no gradient)."""
    x_arr = np.array([x["core"], x["edge"]])
    return {"energy": six_hump_camel_func(x_arr)}


def sim_func_f_and_grad(x):
    """Six-hump camel: return objective and gradient."""
    x_arr = np.array([x["core"], x["edge"]])
    f = six_hump_camel_func(x_arr)
    g = six_hump_camel_grad(x_arr)
    return {"energy": f, "grad_core": g[0], "grad_edge": g[1]}


def sim_func_highdim(x):
    """Six-hump camel on high-dimensional input: use only the first two components."""
    x_arr = np.array([x["x0"], x["x1"]])
    return {"energy": six_hump_camel_func(x_arr)}


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    workflow = Ensemble(parse_args=True)

    if workflow.is_manager:
        start_time = time()

    n = 2

    vocs = VOCS(
        variables={
            "core": [-3, 3],
            "edge": [-2, 2],
            "core_on_cube": [0, 1],
            "edge_on_cube": [0, 1],
        },
        objectives={"energy": "MINIMIZE"},
    )

    variables_mapping_base = {
        "x": ["core", "edge"],
        "x_on_cube": ["core_on_cube", "edge_on_cube"],
        "f": ["energy"],
    }

    exit_criteria = ExitCriteria(sim_max=1000)

    for run in range(2):
        if run == 0:
            # Run 0: scipy Nelder-Mead (derivative-free)
            aposmm = APOSMM(
                vocs,
                max_active_runs=6,
                variables_mapping=variables_mapping_base,
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
                batch_size=5,
                initial_batch_size=100,
            )
            workflow.sim_specs = SimSpecs(simulator=sim_func_f_only, vocs=vocs)

        else:
            # Run 1: scipy BFGS (gradient-based)
            # VOCS needs gradient output variables so sim can return grad components
            vocs_with_grad = VOCS(
                variables={
                    "core": [-3, 3],
                    "edge": [-2, 2],
                    "core_on_cube": [0, 1],
                    "edge_on_cube": [0, 1],
                },
                objectives={"energy": "MINIMIZE"},
                observables=["grad_core", "grad_edge"],
            )

            variables_mapping_bfgs = {
                "x": ["core", "edge"],
                "x_on_cube": ["core_on_cube", "edge_on_cube"],
                "f": ["energy"],
                "grad": ["grad_core", "grad_edge"],
            }

            aposmm_bfgs = APOSMM(
                vocs_with_grad,
                max_active_runs=6,
                variables_mapping=variables_mapping_bfgs,
                initial_sample_size=100,
                sample_points=np.round(minima, 1),
                localopt_method="scipy_BFGS",
                opt_return_codes=[0],
                nu=1e-8,
                mu=1e-8,
                dist_to_bound_multiple=0.01,
            )

            workflow.gen_specs = GenSpecs(
                generator=aposmm_bfgs,
                vocs=vocs_with_grad,
                batch_size=5,
                initial_batch_size=100,
            )
            workflow.sim_specs = SimSpecs(simulator=sim_func_f_and_grad, vocs=vocs_with_grad)

        workflow.exit_criteria = exit_criteria

        H, persis_info, flag = workflow.run()

        if workflow.is_manager:
            print(f"[Manager] Run {run}:", H[np.where(H["local_min"])]["x"])
            print("[Manager]: Time taken =", time() - start_time, flush=True)

            tol = 1e-3
            min_found = 0
            for m in minima:
                print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
                if np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol:
                    min_found += 1
            assert min_found >= 2, f"Run {run}: Found {min_found} minima"

    # High-dimensional run (n=400): test scaling, don't check convergence
    # Build a VOCS with 400 variables
    n_hd = 400
    hd_vars = {f"x{i}": [-1, 1] for i in range(n_hd)}
    hd_vars["x0"] = [-3, 3]
    hd_vars["x1"] = [-2, 2]
    hd_cube_vars = {f"x{i}_cube": [0, 1] for i in range(n_hd)}
    vocs_hd = VOCS(
        variables={**hd_vars, **hd_cube_vars},
        objectives={"energy": "MINIMIZE"},
    )

    variables_mapping_hd = {
        "x": [f"x{i}" for i in range(n_hd)],
        "x_on_cube": [f"x{i}_cube" for i in range(n_hd)],
        "f": ["energy"],
    }

    aposmm_hd = APOSMM(
        vocs_hd,
        max_active_runs=6,
        variables_mapping=variables_mapping_hd,
        initial_sample_size=100,
        localopt_method="scipy_Nelder-Mead",
        opt_return_codes=[0],
        rk_const=4.90247,
    )

    workflow.gen_specs = GenSpecs(
        generator=aposmm_hd,
        vocs=vocs_hd,
        batch_size=5,
        initial_batch_size=100,
    )
    workflow.sim_specs = SimSpecs(simulator=sim_func_highdim, vocs=vocs_hd)
    workflow.exit_criteria = ExitCriteria(sim_max=1000)

    H, persis_info, flag = workflow.run()

    if workflow.is_manager:
        assert np.sum(H["sim_ended"]) >= 990, "Not enough high-dim runs finished"
