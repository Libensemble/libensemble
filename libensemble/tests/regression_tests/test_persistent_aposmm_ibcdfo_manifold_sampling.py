"""
Runs libEnsemble with APOSMM+IBCDFO on two test problems. Only a single
optimization run is being performed for the below setup.

The first case uses POUNDERS to solve the chwirut least-squares problem. For
this case, all chwirut 214 residual calculations for a given point are
performed as a single simulation evaluation.

The second case uses the generalized POUNDERS to minimize normalized beamline
emittance. The "beamline simulation" is a synthetic polynomial test function
that takes in 4 variables and returning 3 outputs. These outputs represent
position <x>, momentum <p_x>, and the correlation between them <x p_x>.

These values are then mapped to the normalized emittance <x> <p_x> - <x p_x>.

Execute via one of the following commands:
   mpiexec -np 3 python test_persistent_aposmm_ibcdfo_pounders.py
   python test_persistent_aposmm_ibcdfo_pounders.py --nworkers 2
Both will run with 1 manager, 1 worker running APOSMM+IBCDFO, and 1 worker
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

libensemble.gen_funcs.rc.aposmm_optimizers = "ibcdfo_manifold_sampling"

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

try:
    from ibcdfo.manifold_sampling import manifold_sampling_primal  # noqa: F401
    from ibcdfo.manifold_sampling.h_examples import pw_maximum as hfun

except ModuleNotFoundError:
    sys.exit("Please 'pip install ibcdfo'")

try:
    from minqsw import minqsw  # noqa: F401

except ModuleNotFoundError:
    sys.exit("Ensure https://github.com/POptUS/minq has been cloned and that minq/py/minq5/ is on the PYTHONPATH")


def synthetic_beamline_mapping(H, _, sim_specs):
    x = H["x"][0]
    assert len(x) == 4, "Assuming 4 inputs to this function"
    y = np.zeros(3)  # Synthetic beamline outputs
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

    nworkers, is_manager, libE_specs, _ = parse_args()

    assert nworkers == 2, "This test is just for two workers"

    m = 3
    n = 4
    sim_f = synthetic_beamline_mapping

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("fvec", float, m)],
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
            "stop_after_k_runs": 1,
            "max_active_runs": 1,
            "sample_points": np.atleast_2d(0.1 * (np.arange(n) + 1)),
            "localopt_method": "ibcdfo_manifold_sampling",
            "run_max_eval": 100 * (n + 1),
            "components": m,
            "lb": -1 * np.ones(n),
            "ub": np.ones(n),
        },
    }

    gen_specs["user"]["hfun"] = hfun

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 500}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert np.min(H["f"]) == 2.0, "The best is 2"
        assert persis_info[1].get("run_order"), "Run_order should have been given back"
        assert flag == 0

        save_libE_output(H, persis_info, __file__, nworkers)
