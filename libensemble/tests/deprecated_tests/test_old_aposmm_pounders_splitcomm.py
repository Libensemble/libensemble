# """
# Runs libEnsemble with APOSMM+POUNDERS on the chwirut least squares problem.
# All 214 residual calculations for a given point are performed as a single
# simulation evaluation. This version uses a split communicator.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python test_chwirut_pounders.py
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 4

import os
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "petsc"
from libensemble.gen_funcs.old_aposmm import aposmm_logic as gen_f

from libensemble.tests.regression_tests.support import persis_info_2 as persis_info, aposmm_gen_out as gen_out
from libensemble.tests.regression_tests.common import mpi_comm_split
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

num_comms = 2  # Must have at least num_comms*2 processors
nworkers, is_manager, libE_specs, _ = parse_args()
libE_specs["mpi_comm"], sub_comm_number = mpi_comm_split(num_comms)
is_manager = libE_specs["mpi_comm"].Get_rank() == 0

# Declare the run parameters/functions
m = 214
n = 3
budget = 10

sim_specs = {
    "sim_f": sim_f,
    "in": ["x"],
    "out": [("f", float), ("fvec", float, m)],
    "user": {"combine_component_func": lambda x: np.sum(np.power(x, 2))},
}

gen_out += [("x", float, n), ("x_on_cube", float, n)]

# lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
gen_specs = {
    "gen_f": gen_f,
    "in": [o[0] for o in gen_out] + ["f", "fvec", "sim_ended"],
    "out": gen_out,
    "user": {
        "initial_sample_size": 5,
        "lb": (-2 - np.pi / 10) * np.ones(n),
        "ub": 2 * np.ones(n),
        "localopt_method": "pounders",
        "dist_to_bound_multiple": 0.5,
        "components": m,
    },
}

gen_specs["user"].update({"grtol": 1e-4, "gatol": 1e-4, "frtol": 1e-15, "fatol": 1e-15})

persis_info = add_unique_random_streams(persis_info, nworkers + 1)

exit_criteria = {"sim_max": budget}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_manager:
    assert flag == 0
    assert len(H) >= budget

    # Calculating the Jacobian at the best point (though this information was not used by pounders)
    from libensemble.sim_funcs.chwirut1 import EvaluateJacobian

    J = EvaluateJacobian(H["x"][np.argmin(H["f"])])
    assert np.linalg.norm(J) < 2000

    outname = os.path.splitext(__file__)[0] + "_sub_comm" + str(sub_comm_number)
    save_libE_output(H, persis_info, outname, nworkers)
