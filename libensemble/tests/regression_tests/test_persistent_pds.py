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
import scipy.optimize as sciopt

from libensemble.libE import libE
from libensemble.sim_funcs.geomedian import geomedian_eval as sim_f
# from libensemble.sim_funcs.alt_rosenbrock import alt_rosenbrock_eval as sim_f
# from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
# from libensemble.sim_funcs.linear_regression import linear_regression_eval as sim_f
# from libensemble.sim_funcs.logistic_regression import logistic_regression_eval as sim_f
# from libensemble.sim_funcs.nesterov_quadratic import nesterov_quadratic_eval as sim_f
from libensemble.gen_funcs.persistent_pds import opt_slide as gen_f
from libensemble.alloc_funcs.start_persistent_consensus import start_consensus_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info
from libensemble.tools.consensus_subroutines import get_k_reach_chain_matrix

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")
if nworkers < 5:
    sys.exit('This tests requires at least 5 workers (6 MPI processes)...')

m = 10
n = 100
num_gens = 4

sim_specs = {'sim_f': sim_f,
             'in': ['x', 'obj_component', 'get_grad'],
             'out': [('f_i', float), ('gradf_i', float, (n,))],
             }

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
                      'lb' : np.zeros(n),
                      'ub' : np.zeros(n),
                      }
             }

alloc_specs = {'alloc_f': alloc_f, 
               'out'    : [], 
               'user'   : {'m': m,
                           'num_gens': num_gens 
                           },
               }

# Problem definition
k = 1
A = spp.diags([1,2,2,1]) - get_k_reach_chain_matrix(num_gens,k)
lam_max = np.amax(la.eig(A.toarray())[0])
np.random.seed(0)

"""
if True: # linear regression
    X = np.array([np.random.normal(loc=0, scale=1.0, size=n) for _ in range(m)]).T
    y = np.dot(X.T, np.ones(n)) + np.cos(np.dot(X.T, np.ones(n))) + np.random.normal(loc=0, scale=0.25, size=m)
    c = 0.1
    X_norms = la.norm(X, ord=2, axis=0)**2

    L = (2/m)*(np.amax(X_norms)+c)
if True: # logistic regression
    y = np.append(2*np.ones(m//2), np.zeros(m-m//2))-1
    X = np.array([np.random.normal(loc=y[i]*np.ones(n), scale=1.0, size=n) for i in range(m)]).T
    c = 0.1

    XXT_sum = np.outer(X[:,0], X[:,0])
    for i in range(1,m):
        XXT_sum += np.outer(X[:,i],X[:,i])
    eig_max = np.amax(la.eig(XXT_sum)[0].real)
    L = eig_max/m
"""
if True: # geometric median
    B = np.random.random((m,n))
    L = 1

    def df(x,i):
        b_i = B[i]
        z = x-b_i
        return (1/m)*z/la.norm(z)
    def f(x,i):
        return (1/m)*la.norm(x-B[i])

eps = 2e-2
persis_info = {}
persis_info['print_progress'] = 0
persis_info['A'] = A

# Include @f_i_eval and @df_i_eval if we want to compute gradient in gen
persis_info['gen_params'] = {
                'mu': 0,      # strong convexity term
                'L': L,       # Lipschitz smoothness
                'Vx_0x': n**0.5, # Bregman divergence of x_0 and x_*
                'eps': eps,   # error / tolerance
                'A_norm': lam_max, # ||A \otimes I||_2 = ||A||_2
                'f_i_eval': f,
                'df_i_eval': df
                }
# persis_info['sim_params'] = { 'X': X, 'y': y, 'c': c }
persis_info['sim_params'] = { 'B': B }
persis_info = add_unique_random_streams(persis_info, nworkers + 1)

exit_criteria = {'elapsed_wallclock_time': 300}

# Perform the run
libE_specs['safe_mode'] = False
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    # check we have a Laplacian matrix
    assert la.norm(A.dot(np.zeros(A.shape[1]))) < 1e-15, 'Not a Laplacian matrix'

    # check we completed
    assert flag == 0

    # compoile sum of {f_i} and {x}, and check their values are bounded by O(eps)
    eval_H = H[H['eval_pt']]
    x = np.empty(n*num_gens, dtype=float)
    F = 0

    gen_ids = np.unique(eval_H['gen_worker'])
    assert len(gen_ids) == num_gens

    for i,gen_id in enumerate(gen_ids):
        last_eval_idx = np.where(eval_H['gen_worker']==gen_id)[0][-1]

        f_i = eval_H[last_eval_idx]['f_i']
        x_i = eval_H[last_eval_idx]['x']

        F += f_i
        x[i*n:(i+1)*n] = x_i

    A_kron_I = spp.kron(A, spp.eye(n))
    consensus_val = np.dot(x, A_kron_I.dot(x))

    # Get fstar
    gtol = eps
    def f_single(x):  return np.sum([f(x,i) for i in range(m)])
    def df_single(x): return np.sum([df(x,i) for i in range(m)], axis=0)
    res = sciopt.minimize(f_single, np.zeros(n), jac=df_single, method="BFGS", tol=eps, options={'gtol': gtol, 'norm': 2, 'maxiter': None})
    fstar = f_single(res.x)

    assert F-fstar < 10*eps, 'Error of {:.4e}, expected {:.4e} (assuming f*={:.4e})'.format(F-fstar, 10*eps, fstar)
    assert consensus_val < eps, 'Consensus score of {:.4e}, expected {:.4e}'.format(consensus_val, eps)
