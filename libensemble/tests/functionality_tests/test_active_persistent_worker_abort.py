"""
Tests libEnsemble capability to abort persistent worker.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_6-hump_camel_active_persistent_worker_abort.py
   python test_6-hump_camel_active_persistent_worker_abort.py --nworkers 3
   python test_6-hump_camel_active_persistent_worker_abort.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4
# TESTSUITE_EXTRA: true

import sys

import numpy as np

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "scipy"

from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens as alloc_f
from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt as gen_f

# Import libEnsemble requirements
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_out += [("x", float, 2), ("x_on_cube", float, 2)]
    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["x", "f"],
        "out": gen_out,
        "user": {
            "localopt_method": "LN_BOBYQA",
            "xtol_rel": 1e-4,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "gen_batch_size": 2,
            "dist_to_bound_multiple": 0.5,
            "localopt_maxeval": 4,
        },
    }

    alloc_specs = {
        "alloc_f": alloc_f,
        "user": {
            "batch_mode": True,
            "num_active_gens": 1,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Set sim_max small so persistent worker is quickly terminated
    exit_criteria = {"sim_max": 10, "wallclock_max": 300}

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert flag == 0
        save_libE_output(H, persis_info, __file__, nworkers)
