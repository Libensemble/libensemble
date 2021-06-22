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

from libensemble.libE import libE
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.gen_funcs.persistent_independent_optimize import independent_optimize as gen_f
from libensemble.alloc_funcs.start_persistent_independent import start_persistent_independent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams 

nworkers, is_manager, libE_specs, _ = parse_args()
if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

m = 16
n = 32
num_gens = 2
assert n==2*m, print("@n must be double of @m")

sim_specs = {'sim_f': sim_f,
             'in': ['x', 'obj_component', 'get_grad'],
             'out': [('f_i', float), ('gradf_i', float, (n,)),
                     ],
             }

# lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,)), 
                     ('pt_id', int),          # which {x_j} to eval
                     ('obj_component', int),  # which {f_i} to eval
                     ('get_grad', bool),
                     ('converged', bool),
                     ('num_f_evals', int),
                     ('num_gradf_evals', int),
                     ],
             'user': {'gen_batch_size': 3,
                      'm': m,
                      'lb' : np.array([-1.2,1]*(n//2)),
                      'ub' : np.array([-1.2,1]*(n//2)),
                      # 'lb' : np.ones(n),
                      # 'ub' : np.ones(n),
                      # 'lb' : 1e-10*np.ones(n),
                      # 'ub': 2*np.ones(n)
                      }
             }

alloc_specs = {'alloc_f': alloc_f, 
               'out'    : [], 
               'user'   : {'num_gens' : num_gens    # number of persistent gens
                           },
               }

persis_info = {}
persis_info['last_H_len'] = 0
persis_info['next_to_give'] = 0
persis_info = add_unique_random_streams(persis_info, nworkers + 1)

# exit_criteria = {'gen_max': 200, 'elapsed_wallclock_time': 300, 'stop_val': ('f', 3000)}
# exit_criteria = {'sim_max': 2000}
exit_criteria = {'elapsed_wallclock_time': 120}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    pass
    # print and save data (don't do that for now)

    # assert len(np.unique(H['gen_time'])) == 10

    # save_libE_output(H, persis_info, __file__, nworkers)
