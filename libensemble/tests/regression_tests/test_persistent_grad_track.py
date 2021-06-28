"""
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

from libensemble.libE import libE
# from libensemble.sim_funcs.chwirut2 import chwirut_eval as sim_f
# from libensemble.sim_funcs.geomedian import geomedian_eval as sim_f
# from libensemble.sim_funcs.convex_funnel import convex_funnel_eval as sim_f
from libensemble.sim_funcs.alt_rosenbrock import alt_rosenbrock_eval as sim_f
# from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.gen_funcs.persistent_grad_track import grad_track as gen_f
from libensemble.alloc_funcs.start_persistent_grad_track import start_gradtrack_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

m = 3  # must match with m in sim_f
n = 4
num_gens = 2

sim_specs = {'sim_f': sim_f,
             'in': ['x', 'obj_component', 'get_grad'],
             'out': [('f_i', float), ('gradf_i', float, (n,))],
             }

# lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,)), 
                     ('consensus_pt', bool),  # does not require a sim
                     ('obj_component', int),  # which {f_i} to eval
                     ('get_grad', bool),
                     ],
             'user': {
                      # 'lb' : -np.ones(n),
                      # 'ub' :  np.ones(n),
                      'lb' : np.array([-1.2,1]*(n//2)),
                      'ub' : np.array([-1.2,1]*(n//2)),
                      }
             }

alloc_specs = {'alloc_f': alloc_f, 
               'out'    : [], 
               'user'   : {'m': m,
                           'num_gens': num_gens 
                           },
               }

persis_info = {}
persis_info['last_H_len'] = 0
persis_info['next_to_give'] = 0
persis_info['hyperparams'] = {
                'R': 10**2,     # penalty (to print ...)
                'L': 10,       # L-smoothness of each function f_i
                'eps': 0.1,     # error / tolerance
                'N_const': 5000,   # multiplicative constant on numiters
                'step_const': 1 # must be <= 1
                }

# hypothesis: things turn awry if finds local minima 

persis_info = add_unique_random_streams(persis_info, nworkers + 1)

# exit_criteria = {'gen_max': 200, 'elapsed_wallclock_time': 300, 'stop_val': ('f', 3000)}
# exit_criteria = {'sim_max': 50000}
exit_criteria = {'elapsed_wallclock_time': 300}

# Perform the run
libE_specs['safe_mode'] = False
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    pass
    # print and save data (don't do that for now)

    # assert len(np.unique(H['gen_time'])) == 10

    # save_libE_output(H, persis_info, __file__, nworkers)
