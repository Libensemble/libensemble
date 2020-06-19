# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_persistent_uniform_sampling.py
#    python3 test_6-hump_camel_persistent_uniform_sampling.py --nworkers 3 --comms local
#    python3 test_6-hump_camel_persistent_uniform_sampling.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.periodic_func import func_wrapper as sim_f

import libensemble.gen_funcs
libensemble.gen_funcs.rc.aposmm_optimizers = 'nlopt'
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams


def assertion(passed):
    """Raise assertion or MPI Abort"""
    if libE_specs['comms'] == 'mpi':
        from mpi4py import MPI
        if passed:
            print("\n\nMPI will be aborted as planned\n\n", flush=True)
            MPI.COMM_WORLD.Abort(0)  # Abort with success
        else:
            MPI.COMM_WORLD.Abort(1)  # Abort with failure
    else:
        assert passed
        print("\n\nException received as expected")


nworkers, is_master, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float)]}

gen_out = [('x', float, n), ('x_on_cube', float, n), ('sim_id', int),
           ('local_min', bool), ('local_pt', bool)]

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': gen_out,
             'user': {'initial_sample_size': 100,
                      'localopt_method': 'LN_BOBYQA',
                      'lb': np.array([0, -np.pi/2]),
                      'ub': np.array([2*np.pi, 3*np.pi/2]),
                      }
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}


exit_criteria = {'sim_max': 1000}

persis_info = add_unique_random_streams({}, nworkers + 1)

libE_specs['abort_on_exception'] = False
try:
    # Perform the run, which will fail because we want to test exception handling
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                alloc_specs, libE_specs)
except Exception as e:
    if is_master:
        if e.args[1] == 'NLopt roundoff-limited':
            assertion(True)
        else:
            assertion(False)
else:
    if is_master:
        assertion(False)
