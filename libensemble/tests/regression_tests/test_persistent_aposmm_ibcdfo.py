"""
Runs libEnsemble with APOSMM+IBCDFO on the chwirut least-squares problem.
All 214 residual calculations for a given point are performed as a single
simulation evaluation.

Execute via one of the following commands:
   mpiexec -np 3 python test_persistent_aposmm_ibcdfo.py
   python test_persistent_aposmm_ibcdfo.py --nworkers 2 --comms local
Both will run with 1 manager, 1 worker running APOSMM+IBCDFO), and 1 worker
doing the simulation evaluations.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 3

import multiprocessing
import sys

import numpy as np

import libensemble.gen_funcs
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f

libensemble.gen_funcs.rc.aposmm_optimizers = "ibcdfo"

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

try:
    from ibcdfo.pounders import pounders  # noqa: F401
except ModuleNotFoundError:
    sys.exit("Please 'pip install ibcdfo'")

try:
    from minqsw import minqsw  # noqa: F401

except ModuleNotFoundError:
    sys.exit("Ensure https://github.com/POptUS/minq has been cloned and that minq/py/minq5/ is on the PYTHONPATH")


def combine_component(x):
    return np.sum(np.power(x, 2))


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

    assert nworkers == 2, "This test is just for two workers"

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

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
        ("started_run", bool),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "fvec"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 1,
            "sample_points": np.array([[0.1, 0.2, 0.345]]),
            "localopt_method": "ibcdfo_pounders",
            "components": m,
            "lb": -2 * np.ones(n),
            "ub": 2 * np.ones(n),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 500}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert persis_info[1].get("run_order"), "Run_order should have been given back"
        assert flag == 0

        save_libE_output(H, persis_info, __file__, nworkers)

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
            #         import ipdb

            #         ipdb.set_trace()
            # else:
            #     d = np.linalg.solve(np.dot(J.T, J), np.dot(J.T, F))
            #     assert np.linalg.norm(d) <= 1e-5
