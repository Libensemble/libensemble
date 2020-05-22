# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_OS_SKIP: OSX

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f, six_hump_camel_func
from libensemble.gen_funcs.persistent_tasmanian import sparse_grid as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
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

gen_specs = {'gen_f': gen_f,
             'in': ['x', 'f'],
             'out': [('x', float, n)],
             'user': {'NumInputs': n,  # Don't need to do evaluations because simulating the sampling already being done
                      'NumOutputs': 1,
                      'x0': np.array([0.3, 0.7]),
                      'precisions': [6, 12]}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

for run in range(2):
    persis_info = add_unique_random_streams({}, nworkers + 1)

    if run == 0:
        exit_criteria = {'elapsed_wallclock_time': 10}
    elif run == 1:
        exit_criteria = {'gen_max': 100}  # This will test persistent_tasmanian stopping early.

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                alloc_specs, libE_specs)

    if is_master:
        if run == 0:
            true_val = six_hump_camel_func(gen_specs['user']['x0'])

            for p in gen_specs['user']['precisions']:
                assert np.abs(true_val - persis_info[1]['aResult'][p]) <= p

            print('[Manager]: Time taken =', time() - start_time, flush=True)

            save_libE_output(H, persis_info, __file__, nworkers)

        if run == 1:
            assert 6 in persis_info[1]['aResult'], "Correctly did this case"
            assert 12 not in persis_info[1]['aResult'], "Correctly stopped short and didn't do this case"
