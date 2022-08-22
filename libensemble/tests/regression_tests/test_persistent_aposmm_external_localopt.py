"""
Test the use for APOSMM with an external local optimization method.
Points are passed to/from the localopt method using files with run-specific
hashes. These hashes are currently generated using uuid, which may not be
thread safe on some systems (e.g., Travis-CI). This was resolved by not using
'local' communication; we therefore recommend using 'mpi' communication when
using persistent_aposmm with an external localopt # method.


Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_aposmm_external_localopt.py

When running with the above command, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 4
# TESTSUITE_OS_SKIP: OSX WIN
# TESTSUITE_EXTRA: true

import sys
import multiprocessing
import numpy as np
import shutil  # For copying the external_localopt script

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from time import time

np.set_printoptions(precision=16)

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    multiprocessing.set_start_method("fork", force=True)

    nworkers, is_manager, libE_specs, _ = parse_args()

    if is_manager:
        start_time = time()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_out = [
        ("x", float, n),
        ("x_on_cube", float, n),
        ("sim_id", int),
        ("local_min", bool),
        ("local_pt", bool),
    ]

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f"] + [n[0] for n in gen_out],
        "out": gen_out,
        "user": {
            "initial_sample_size": 100,
            "sample_points": np.round(minima, 1),
            "localopt_method": "external_localopt",
            "max_active_runs": 6,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }
    shutil.copy("./scripts_used_by_reg_tests/call_matlab_octave_script.m", "./")
    shutil.copy("./scripts_used_by_reg_tests/wrapper_obj_fun.m", "./")

    alloc_specs = {"alloc_f": alloc_f}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 500}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        print("[Manager]:", H[np.where(H["local_min"])]["x"])
        print("[Manager]: Time taken =", time() - start_time, flush=True)

        # Note: This regression test considers only the global minima because it's
        # not possible to pass an initial simplex to fminsearch in Octave/Matlab.
        # As such, localopt runs in APOSMM that are started near 4 of the 6 local
        # minima "jump out", even when 'sim_max' is increased. (Matlab's fminsearch
        # has a smaller initial simplex and appears to be less susceptible to
        # this.)
        minima = minima[:2]
        tol = 1e-3
        for m in minima:
            # The minima are known on this test problem.
            # We use their values to test APOSMM has identified all minima
            print(np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)), flush=True)
            assert np.min(np.sum((H[H["local_min"]]["x"] - m) ** 2, 1)) < tol

        save_libE_output(H, persis_info, __file__, nworkers)
