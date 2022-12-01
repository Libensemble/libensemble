# """
# Runs libEnsemble with APOSMM+POUNDERS on the chwirut least squares problem.
# Each of the 214 residual calculation for a given point is performed as a
# separate simulation evaluation.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python test_chwirut_aposmm_one_residual_at_a_time.py
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "petsc"
from libensemble.gen_funcs.old_aposmm import aposmm_logic as gen_f

from libensemble.alloc_funcs.fast_alloc_and_pausing import give_sim_work_first as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info, aposmm_gen_out as gen_out

nworkers, is_manager, libE_specs, _ = parse_args()

# Declare the run parameters/functions
m = 214
n = 3
budget = 50 * m

sim_specs = {
    "sim_f": sim_f,
    "in": ["x", "obj_component"],
    "out": [("f_i", float)],
}

gen_out += [("x", float, n), ("x_on_cube", float, n), ("obj_component", int), ("f", float)]

# LB tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
UB = 2 * np.ones(n)
LB = (-2 - np.pi / 10) * np.ones(n)
gen_specs = {
    "gen_f": gen_f,
    "in": [o[0] for o in gen_out] + ["f_i", "sim_ended"],
    "out": gen_out,
    "user": {
        "initial_sample_size": 5,
        "lb": LB,
        "ub": UB,
        "localopt_method": "pounders",
        "dist_to_bound_multiple": 0.5,
        "single_component_at_a_time": True,
        "components": m,
        "combine_component_func": lambda x: np.sum(np.power(x, 2)),
    },
}

gen_specs["user"].update({"grtol": 1e-4, "gatol": 1e-4, "frtol": 1e-15, "fatol": 1e-15})

np.random.seed(0)
gen_specs["user"]["sample_points"] = np.random.uniform(0, 1, (budget, n)) * (UB - LB) + LB
alloc_specs = {
    "alloc_f": alloc_f,
    "out": [],
    "user": {
        "stop_on_NaNs": True,
        "batch_mode": True,
        "num_active_gens": 1,
        "stop_partial_fvec_eval": True,
    },
}

persis_info = add_unique_random_streams(persis_info, nworkers + 1)

exit_criteria = {"sim_max": budget}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

if is_manager:
    assert flag == 0
    assert len(H) >= budget
    save_libE_output(H, persis_info, __file__, nworkers)
