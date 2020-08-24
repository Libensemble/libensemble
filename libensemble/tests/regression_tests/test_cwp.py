# """
# Runs libEnsemble with cwp calibration test.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-test_cwp.py
#    python3 test_6-test_cwp.py --nworkers 3 --comms local
#    python3 test_6-test_cwp.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

# NOTE (REMOVE WHEN FIXED. CURRENTLY NOT WORKING - RUNS SIM FUNCS AND ERRORS IN emulation_builder.

# Requires:
#   Clone cwpcalibration repo
#   pip install functionbase package.

# TODO for step 1:
#    Determine exit_criteria (currently returns to gen but does not finish).
#    Determine output from model.
#    ---Create valid sim func / problem definition---
#    ---What is A/B in x array - I dont have this.--- Used 0/1 for now.
#    Rename files/vars as required.
#    Decide if we want persistent gen / persistence of x/thetas
#    Determine pass condition for test (assertions at end).

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.gen_funcs.cwp import testmseerror as gen_f
from libensemble.sim_funcs.cwpsim import borehole as sim_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams


if __name__ == '__main__':

    nworkers, is_master, libE_specs, _ = parse_args()

    sim_specs = {'sim_f': sim_f, 'in': ['x', 'thetas'], 'out': [('f', float)]}

    n_thetas = 10  # 100
    n_x = 5        # 50
    nparams = 6    # No. of theta params
    ndims = 3      # No. of x co-ordinates.

    n_evals = n_thetas * n_x

    gen_out = [('x', float, ndims), ('thetas', float, nparams), ('mse', float, (1,))]
    gen_specs = {'gen_f': gen_f,
                 'in': [o[0] for o in gen_out]+['f', 'returned'],
                 'out': gen_out,
                 'user': {'n_thetas': n_thetas,    # Num thetas
                          'gen_batch_size': n_x,   # Num x points to create
                          'lb': np.array([0, 0]),  # Low bound for x
                          'ub': np.array([6, 6]),  # High bound for x
                          }
                 }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    # exit_criteria = {'sim_max': n_evals}
    exit_criteria = {'sim_max': n_evals + n_x, # Evaluate at n_x more points, i.e. one additional theta (row)
                     'stop_val': ('mse', 10 ** (-4))} # stop when mse is less than a threshold

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs,
                                exit_criteria, persis_info,
                                libE_specs=libE_specs)

    if is_master:
        assert np.all(H['returned'])
        save_libE_output(H, persis_info, __file__, nworkers)
