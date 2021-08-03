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
import numpy.linalg as la
import scipy.sparse as spp

from libensemble.libE import libE
# from libensemble.sim_funcs.chwirut2 import chwirut_eval as sim_f
from libensemble.sim_funcs.geomedian import geomedian_eval as sim_f
# from libensemble.sim_funcs.convex_funnel import convex_funnel_eval as sim_f
# from libensemble.sim_funcs.alt_rosenbrock import alt_rosenbrock_eval as sim_f
# from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.gen_funcs.persistent_prox_slide import opt_slide as gen_f
from libensemble.alloc_funcs.start_persistent_consensus import start_consensus_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info

def get_k_reach_chain_matrix(n, k):
    """ Constructs adjacency matrix for a chain matrix where the ith vertex can
        reach vertices that are at most @k distances from them (does not wrap around),
        where the distance is based on the absoluate difference between vertices'
        indexes.
    """
    assert 1 <= k <= n-1

    half_of_diagonals = [np.ones(n-k+j) for j in range(k)]
    half_of_indices = np.arange(1,k+1)
    all_of_diagonals = half_of_diagonals + half_of_diagonals[::-1]
    all_of_indices = np.append(-half_of_indices[::-1], half_of_indices)
    A = spp.csr_matrix( spp.diags(all_of_diagonals, all_of_indices) )
    return A

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

m = 10  # must match with m in sim_f
n = 10
num_gens = 3

sim_specs = {'sim_f': sim_f,
             'in': ['x', 'obj_component', 'get_grad'],
             'out': [('f_i', float), ('gradf_i', float, (n,))],
             }

# lb tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,)), 
                     ('f_i', float),
                     ('eval_pt', bool),       # eval point 
                     ('consensus_pt', bool),  # does not require a sim
                     ('obj_component', int),  # which {f_i} to eval
                     ('get_grad', bool),
                     ],
             'user': {
                      'lb' : -np.ones(n),
                      'ub' : np.ones(n),
                      }
             }

# TODO: Do we need ret_to_gen?
alloc_specs = {'alloc_f': alloc_f, 
               'out'    : [], 
               'user'   : {'m': m,
                           'num_gens': num_gens 
                           },
               }

k = 1
A = spp.diags([1,2,1])  - get_k_reach_chain_matrix(num_gens,k)
lam_max = np.amax(la.eig(A.toarray())[0])

persis_info = {}
persis_info['print_progress'] = 1
persis_info['A'] = A
persis_info['params'] = {
                'M': 1,       # upper bound on gradient
                'R': 10**2,   # consensus penalty
                'nu': 1,      # modulus of strongly convex DGF 
                'eps': 0.1,   # error / tolerance
                'D': 2*n,     # diameter of domain
                'N_const': 1, # multiplicative constant on numiters
                'lam_max': lam_max # ||A \otimes I||_2 = ||A||_2
                }

persis_info = add_unique_random_streams(persis_info, nworkers + 1)

# exit_criteria = {'gen_max': 200, 'elapsed_wallclock_time': 300, 'stop_val': ('f', 3000)}
# exit_criteria = {'sim_max': 50000}
exit_criteria = {'elapsed_wallclock_time': 300}

# Perform the run
libE_specs['safe_mode'] = False
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

