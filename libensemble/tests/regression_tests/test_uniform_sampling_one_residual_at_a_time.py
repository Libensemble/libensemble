"""
Runs libEnsemble with uniform random sampling on the chwirut least squares
problem.  All 214 residual calculations for a given point are performed as a
single simulation evaluation. NaNs are injected probabilistically in order to
test the allocation function's ability to preempt future residual
calculations. Also, the allocation function tries to preempt calculations
corresponding to points with partial sum-squared error worse than the
best-evaluated point so far.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_uniform_sampling_one_residual_at_a_time.py

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4

import sys
import numpy as np
from copy import deepcopy

# Import libEnsemble items
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample_obj_components as gen_f
from libensemble.alloc_funcs.fast_alloc_and_pausing import give_sim_work_first
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()
    if libE_specs["comms"] == "tcp":
        # Can't use the same interface for manager and worker if we want
        # repeated calls to libE -- the manager sets up a different server
        # each time, and the worker will not know what port to connect to.
        sys.exit("Cannot run with tcp when repeated calls to libE -- aborting...")

    # Declare the run parameters/functions
    m = 214
    n = 3
    budget = 10 * m

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x", "obj_component"],
        "out": [("f_i", float)],
        "user": {"component_nan_frequency": 0.01},
    }

    # lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
    gen_specs = {
        "gen_f": gen_f,
        "in": ["pt_id"],
        "out": [("x", float, n), ("priority", float), ("paused", bool), ("obj_component", int), ("pt_id", int)],
        "user": {
            "gen_batch_size": 2,
            "single_component_at_a_time": True,
            "combine_component_func": lambda x: np.sum(np.power(x, 2)),
            "lb": (-2 - np.pi / 10) * np.ones(n),
            "ub": 2 * np.ones(n),
            "components": m,
        },
    }

    alloc_specs = {
        "alloc_f": give_sim_work_first,  # Allocation function
        "out": [],  # Output fields (included in History)
        "user": {
            "stop_on_NaNs": True,  # Should alloc preempt evals
            "batch_mode": True,  # Wait until all sim evals are done
            "num_active_gens": 1,  # Only allow one active generator
            "stop_partial_fvec_eval": True,  # Should alloc preempt evals
        },
    }
    # end_alloc_specs_rst_tag

    persis_info = add_unique_random_streams(persis_info, nworkers + 1)
    persis_info_safe = deepcopy(persis_info)

    exit_criteria = {"sim_max": budget, "wallclock_max": 300}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
    if is_manager:
        assert flag == 0
        save_libE_output(H, persis_info, __file__, nworkers)

    # Perform the run but not stopping on NaNs
    alloc_specs["user"].pop("stop_on_NaNs")
    persis_info = deepcopy(persis_info_safe)
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
    if is_manager:
        assert flag == 0

    # Perform the run also not stopping on partial fvec evals
    alloc_specs["user"].pop("stop_partial_fvec_eval")
    persis_info = deepcopy(persis_info_safe)
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
    if is_manager:
        assert flag == 0
