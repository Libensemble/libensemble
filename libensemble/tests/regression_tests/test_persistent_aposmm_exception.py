"""
Tests the APOSMM generator function's ability to handle exceptions

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_exception.py
   python test_persistent_aposmm_exception.py --nworkers 3

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import multiprocessing
import sys

import numpy as np

import libensemble.gen_funcs

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.periodic_func import func_wrapper as sim_f

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.tools import add_unique_random_streams, parse_args


def assertion(passed):
    """Raise assertion or MPI Abort"""
    if libE_specs["comms"] == "mpi":
        from mpi4py import MPI

        if passed:
            print("\n\nMPI will be aborted as planned\n\n", flush=True)
            MPI.COMM_WORLD.Abort(0)  # Abort with success
        else:
            MPI.COMM_WORLD.Abort(1)  # Abort with failure
    else:
        assert passed
        print("\n\nException received as expected")


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_out = [("x", float, n), ("x_on_cube", float, n), ("sim_id", int), ("local_min", bool), ("local_pt", bool)]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "localopt_method": "LN_BOBYQA",
            "lb": np.array([0, -np.pi / 2]),
            "ub": np.array([2 * np.pi, 3 * np.pi / 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    exit_criteria = {"sim_max": 1000}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    libE_specs["abort_on_exception"] = False
    try:
        # Perform the run, which will fail because we want to test exception handling
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
    except Exception as e:
        if is_manager:
            if e.args[1].endswith("NLopt roundoff-limited"):
                assertion(True)
            else:
                assertion(False)
    else:
        if is_manager:
            assertion(False)
