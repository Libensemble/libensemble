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
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
# from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.sim_funcs.chwirut2 import chwirut_eval as sim_f
# from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform as gen_f
from libensemble.gen_funcs.persistent_smart_sampling import persistent_smart as gen_f
from libensemble.alloc_funcs.start_smart_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

m = 214
n = 3

sim_specs = {'sim_f': sim_f,
             'in': ['x', 'obj_component'],
             'out': [('f_i', float)],
             }

# lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,)), 
                     ('priority', float),
                     ('pt_id', int),          # which {x_j} to eval
                     ('obj_component', int)], # which {f_i} to eval
             'user': {'gen_batch_size': 3,
                      'm': m,
                      'combine_component_func': lambda x : np.sum(x), 
                      'lb': (-2-np.pi/10)*np.ones(n),
                      'ub': 2*np.ones(n)}
             }

alloc_specs = {'alloc_f': alloc_f, 
               'out'    : [('given_back', bool)],
               'user'   : {'stop_partial_eval' : True,
                           'num_gens'          : 2    # number of persistent gens
                           },
               }

persis_info = add_unique_random_streams({}, nworkers + 1)

# exit_criteria = {'gen_max': 200, 'elapsed_wallclock_time': 300, 'stop_val': ('f', 3000)}
exit_criteria = {'sim_max': 200}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    pass
    # print and save data (don't do that for now)

    # assert len(np.unique(H['gen_time'])) == 10

    # save_libE_output(H, persis_info, __file__, nworkers)
