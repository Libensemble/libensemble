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
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

import libensemble.gen_funcs
libensemble.gen_funcs.rc.aposmm_optimizers = 'scipy'
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from time import time

nworkers, is_master, libE_specs, _ = parse_args()

if is_master:
    start_time = time()

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
                      'sample_points': np.round(minima, 1),
                      'localopt_method': 'scipy_Nelder-Mead',
                      'opt_return_codes': [0],
                      'nu': 1e-8,
                      'mu': 1e-8,
                      'dist_to_bound_multiple': 0.01,
                      'max_active_runs': 6,
                      'lb': np.array([-3, -2]),
                      'ub': np.array([3, 2])}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}


exit_criteria = {'sim_max': 2000}


for run in range(2):
    persis_info = add_unique_random_streams({}, nworkers + 1)

    if run == 1:
        gen_specs['user']['localopt_method'] = 'scipy_BFGS'
        gen_specs['user']['opt_return_codes'] = [0]
        sim_specs['out'] = [('f', float), ('grad', float, n)]

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                alloc_specs, libE_specs)

    if is_master:
        print('[Manager]:', H[np.where(H['local_min'])]['x'])
        print('[Manager]: Time taken =', time() - start_time, flush=True)

        tol = 1e-3
        min_found = 0
        for m in minima:
            # The minima are known on this test problem.
            # We use their values to test APOSMM has identified all minima
            print(np.min(np.sum((H[H['local_min']]['x'] - m)**2, 1)), flush=True)
            if np.min(np.sum((H[H['local_min']]['x'] - m)**2, 1)) < tol:
                min_found += 1
        assert min_found >= 4, "Found {} minima".format(min_found)

        save_libE_output(H, persis_info, __file__, nworkers)
