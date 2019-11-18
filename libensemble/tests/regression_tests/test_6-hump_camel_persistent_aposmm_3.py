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
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from math import gamma, pi, sqrt
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.utils import parse_args, save_libE_output, add_unique_random_streams
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
             'out': [('f', float), ('grad', float, n)]}

gen_out = [('x', float, n), ('x_on_cube', float, n), ('sim_id', int),
           ('local_min', bool), ('local_pt', bool)]

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': gen_out,
             'user': {'initial_sample_size': 100,
                      'sample_points': np.round(minima, 1),
                      'localopt_method': 'blmvm',
                      'rk_const': 0.5*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
                      'grtol': 1e-4,
                      'gatol': 1e-4,
                      'num_active_gens': 1,
                      'dist_to_bound_multiple': 0.5,
                      'max_active_runs': 6,
                      'lb': np.array([-3, -2]),
                      'ub': np.array([3, 2])}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {'batch_mode': True}}

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': 1000}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    print('[Manager]:', H[np.where(H['local_min'])]['x'])
    print('[Manager]: Time taken =', time() - start_time, flush=True)

    tol = 1e-5
    for m in minima:
        # The minima are known on this test problem.
        # We use their values to test APOSMM has identified all minima
        print(np.min(np.sum((H[H['local_min']]['x'] - m)**2, 1)), flush=True)
        assert np.min(np.sum((H[H['local_min']]['x'] - m)**2, 1)) < tol

    save_libE_output(H, persis_info, __file__, nworkers)
