"""
# Runs libEnsemble on variety of smooth convex terms, including:
#    - Rosenbrock function
#    - Alternative Rosenbrock function
#    - Nesterov's quadratic function (Sec 2.1.2 in Nesterov's 's "Introductory Lectures on Convex Programming")
#    - Linear regression with l2 regularization
#    - Logistic regression with l2 regularization
#    - PYCUTEst functions (https://jfowkes.github.io/pycutest/_build/html/index.html)
#
# You can specify which problem to test by setting @prob_id \in {0,1,2,3,4,5}. 
#
# This call script uses distributed gradient-tracking (https://ieeexplore.ieee.org/iel7/6509490/6730648/09199106.pdf?casa_token=7Z9O3Cf1TrwAAAAA:rSWhlWmhirmn5BLzEEbxGOQh-KCbpjrwDPHU1ugmfoY6pXsTQSNcurJjAsz4SH2azvbucJC-m1o)
# to solve the following problems. To test, run using, for any p >= 6,
#    mpiexec -np p python3 test_persistent_n_agent.py
#    python3 test_persistent_n_agent.py --nworkers p --comms local
#    python3 test_persistent_n_agent.py --nworkers p --comms tcp
#
# The number gens will be 4. 
# 
# Note that this library makes use the CVXPY library to obtain the optimal value
# for regression testing. To test PYCUTEst, make sure to install the necessary
# files. Refer to tools/pycute_interface on more details. To ignore this library,
# simply comment out the import of "Blackbox" below
# """

import numpy as np
import numpy.linalg as la
import scipy.sparse as spp

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

from libensemble.libE import libE
from libensemble.gen_funcs.persistent_n_agent import n_agent as gen_f
from libensemble.alloc_funcs.start_persistent_consensus import start_consensus_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info
from libensemble.tools.consensus_subroutines import get_k_reach_chain_matrix, get_doubly_stochastic, regls_opt, log_opt
from libensemble.tools.pycute_interface import Blackbox

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")
if nworkers < 5:
    sys.exit('This tests requires at least 5 workers (6 MPI processes)...')

num_gens = 4
A = spp.diags([2,3,3,2]) - get_k_reach_chain_matrix(num_gens,2)
S = get_doubly_stochastic(A)
rho = la.norm(S - (S.shape[0]**-1)*np.ones(S.shape), ord=2)

eps = 1e-1
persis_info = {}
persis_info['print_progress'] = 0
persis_info['A'] = S
N_const = 100

persis_info = add_unique_random_streams(persis_info, nworkers + 1)
persis_info['gen_params'] = {}
exit_criteria = {'elapsed_wallclock_time': 300}

# Perform the run
libE_specs['safe_mode'] = False

# 0: rosenbrock, 1: alt rosenbrock, 2: nesterov's, 3: l2 linear regression, 4: l2 logistic regression, 5: CUTEr
prob_id = 4

if prob_id == 0:
    from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
    m,n = 10,20
    prob_name = 'Chained Rosenbrock'
    L = 1
    fstar = 0
    err_const = 1e3

if prob_id == 1:
    from libensemble.sim_funcs.alt_rosenbrock import alt_rosenbrock_eval as sim_f
    m,n = 14,15
    prob_name = 'Alternative chained Rosenbrock'
    L = 1
    fstar = 0
    err_const = 1e3

if prob_id == 2:
    from libensemble.sim_funcs.nesterov_quadratic import nesterov_quadratic_eval as sim_f
    m,n = 15,14
    prob_name = "Nesterov's quadratic function"
    L = 4
    # See Sec 2.1.2 of Nesterov's "Introductory Lectures on Convex Programming"
    fstar = 0.5*(-1+1/(m+1)) 
    err_const = 1

if prob_id == 3:
    from libensemble.sim_funcs.linear_regression import linear_regression_eval as sim_f
    m,n = 4,15
    prob_name = 'linear regression with l2 regularization'
    L = 1
    err_const = 1e1
    X = np.array([np.random.normal(loc=0, scale=1.0, size=n) for _ in range(m)]).T
    y = np.dot(X.T, np.ones(n)) + np.cos(np.dot(X.T, np.ones(n))) + np.random.normal(loc=0, scale=0.25, size=m)
    c = 0.1

    X_norms = la.norm(X, ord=2, axis=0)**2
    L = (2/m)*(np.amax(X_norms)+c)

    # reduce size of problem to match avaible gens
    persis_info['sim_params'] = {'X': X, 'y': y, 'c': c, 'reg': 'l2'}
    fstar = regls_opt(X, y, c, reg='l2')

    def df(theta,i):
        return (2/m)*(-y[i]+np.dot(X[:,i],theta)) * X[:,i] + (2*c/m)*theta

    def f(theta,i):
        z = y[i] - np.dot(X[:,i], theta)
        return (1/m)*np.dot(z,z) + (c/m)*np.dot(theta,theta)

    # Setting @f_i_eval and @df_i_eval tells to gen to compute gradients locally
    persis_info['gen_params'] = { 'f_i_eval': f, 'df_i_eval': df }

if prob_id == 4:
    from libensemble.sim_funcs.logistic_regression import logistic_regression_eval as sim_f
    m,n = 14,15
    prob_name = 'logistic regression  with l2 regularization'
    L = 1
    err_const = 1e1
    y = np.append(2*np.ones(m//2), np.zeros(m-m//2))-1
    X = np.array([np.random.normal(loc=y[i]*np.ones(n), scale=1.0, size=n) for i in range(m)]).T
    c = 0.1

    XXT_sum = np.outer(X[:,0], X[:,0])
    for i in range(1,m):
        XXT_sum += np.outer(X[:,i],X[:,i])
    eig_max = np.amax(la.eig(XXT_sum)[0].real)
    L = eig_max/m

    persis_info['sim_params'] = {'X': X, 'y': y, 'c': c, 'reg': 'l2'}
    fstar = log_opt(X, y, c, 'l2')

elif prob_id == 5:
    from libensemble.sim_funcs.pycute import pycute_eval as sim_f
    n = 100
    m = 4
    prob_name = 'PYCUTEST function'

    bbox = Blackbox(k=m)
    bbox.setup_new_prob(seed_num=0)
    bbox.set_scale()
    L = 1
    N_const = 10

    err_const = 1e2
    persis_info['sim_params'] = {'m': m}
    [fstar, _] = bbox.get_optimal_sol()
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
                      'lb' : -np.ones(n),
                      'ub' : np.ones(n),
                      }
             }

alloc_specs = {'alloc_f': alloc_f, 
               'out'    : [], 
               'user'   : {'m': m,
                           'num_gens': num_gens 
                           },
               }

# Include @f_i_eval and @df_i_eval if we want to compute gradient in gen
persis_info['gen_params'].update({
                'L': 1,       # L-smoothness of each function f_i
                'eps': eps,     # error / tolerance
                'rho': rho, 
                'N_const': N_const,   # multiplicative constant on numiters
                'step_const': 1
                })

if is_manager: print('=== Optimizing {} ==='.format(prob_name), flush=True)

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                        alloc_specs, libE_specs)

if is_manager: print('=== End algorithm ===', flush=True)

if is_manager:
    # check we completed
    assert flag == 0

    # check we have a Laplacian matrix
    assert la.norm(A.dot(np.zeros(A.shape[1]))) < 1e-15, 'Not a Laplacian matrix'

    # check we have a doubly stochastic matrix
    assert la.norm(np.ones(num_gens) - S.dot(np.ones(num_gens)))/num_gens**0.5 < 1e-15, '@S is not a doubly stochastic matrix'

    # compile sum of {f_i} and {x}, and check their values are bounded by O(eps)
    eval_H = H[H['eval_pt']]

    gen_ids = np.unique(eval_H['gen_worker'])
    assert len(gen_ids) == num_gens, 'Gen did not submit any function eval requests'

    x = np.empty(n*num_gens, dtype=float)
    F = 0

    for i,gen_id in enumerate(gen_ids):
        last_eval_idx = np.where(eval_H['gen_worker']==gen_id)[0][-1]

        f_i = eval_H[last_eval_idx]['f_i']
        x_i = eval_H[last_eval_idx]['x']

        F += f_i
        x[i*n:(i+1)*n] = x_i

    A_kron_I = spp.kron(A, spp.eye(n))
    consensus_val = np.dot(x, A_kron_I.dot(x))

    assert F-fstar < err_const*eps, 'Error of {:.4e}, expected {:.4e} (assuming f*={:.4e})'.format(F-fstar, err_const*eps, fstar)
    assert consensus_val < eps, 'Consensus score of {:.4e}, expected {:.4e}\nx={}'.format(consensus_val, eps, x)
