# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f, six_hump_camel_func
from libensemble.gen_funcs.persistent_tasmanian import sparse_grid  as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.utils import parse_args, save_libE_output, add_unique_random_streams
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
                      'precisions': [6,12]}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'elapsed_wallclock_time': 10}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    # import ipdb; ipdb.set_trace()
    # six_hump_camel_func(gen_specs['x0'])
    
    print('[Manager]:', H[np.where(H['local_min'])]['x'])
    print('[Manager]: Time taken =', time() - start_time, flush=True)

    save_libE_output(H, persis_info, __file__, nworkers)
