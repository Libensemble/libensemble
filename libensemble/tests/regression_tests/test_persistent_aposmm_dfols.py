"""
Runs libEnsemble with APOSMM+DFOLS on the chwirut least-squares problem.
All 214 residual calculations for a given point are performed as a single
simulation evaluation.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_dfols.py
   python test_persistent_aposmm_dfols.py --nworkers 3
   python test_persistent_aposmm_dfols.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
import sys

import numpy as np

import libensemble.gen_funcs

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f

libensemble.gen_funcs.rc.aposmm_optimizers = "dfols"

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output


def combine_component(x):
    return np.sum(np.power(x, 2))


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    # Declare the run parameters/functions
    m = 214
    n = 3

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("fvec", float, m)],
        "user": {
            "combine_component_func": combine_component,
        },
    }

    gen_out = [("x", float, n), ("x_on_cube", float, n), ("sim_id", int), ("local_min", bool), ("local_pt", bool)]

    # lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "fvec"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "localopt_method": "dfols",
            "components": m,
            "dfols_kwargs": {
                "do_logging": False,
                "rhoend": 1e-5,
                "user_params": {
                    "model.abs_tol": 1e-10,
                    "model.rel_tol": 1e-4,
                },
            },
            "lb": (-2 - np.pi / 10) * np.ones(n),
            "ub": 2 * np.ones(n),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Tell libEnsemble when to stop (stop_val key must be in H)
    exit_criteria = {
        "sim_max": 1000,
        "wallclock_max": 100,
        "stop_val": ("f", 3000),
    }
    # end_exit_criteria_rst_tag

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert persis_info[1].get("run_order"), "Run_order should have been given back"
        assert flag == 0
        assert np.min(H["f"][H["sim_ended"]]) <= 3000, "Didn't find a value below 3000"

        save_libE_output(H, persis_info, __file__, nworkers)

        # # Calculating the Jacobian at local_minima (though this information was not used by DFO-LS)
        from libensemble.sim_funcs.chwirut1 import EvaluateFunction, EvaluateJacobian

        for i in np.where(H["local_min"])[0]:
            F = EvaluateFunction(H["x"][i])
            J = EvaluateJacobian(H["x"][i])
            # u = gen_specs["user"]["ub"] - H["x"][i]
            # l = H["x"][i] - gen_specs["user"]["lb"]
            # if np.any(u <= 1e-7) or np.any(l <= 1e-7):
            #     grad = -2 * np.dot(J.T, F)
            #     assert np.all(grad[u <= 1e-7] >= 0)
            #     assert np.all(grad[l <= 1e-7] <= 0)

            #     if not np.all(grad[np.logical_and(u >= 1e-7, l >= 1e-7)] <= 1e-5):
            # else:
            #     d = np.linalg.solve(np.dot(J.T, J), np.dot(J.T, F))
            #     assert np.linalg.norm(d) <= 1e-5

    if libE_specs["comms"] == "mpi":
        # Quickly try a different DFO-LS exit condition
        persis_info = add_unique_random_streams({}, nworkers + 1)
        gen_specs["user"]["dfols_kwargs"]["rhoend"] = 1e-16
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            assert flag == 0
