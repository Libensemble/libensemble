"""
Runs libEnsemble on variety of non-smooth convex terms, including:
   - Geometric median
   - SVM with l1 regularization

You can specify which problem to test by setting @prob_id in {0,1}.

This call script uses proximal gradient sliding (https://doi.org/10.1007/s10107-015-0955-5)
to solve the following problems. To test, run using, for any p >= 6,
   mpiexec -np p python3 test_persistent_prox_slide.py
   python3 test_persistent_prox_slide.py --nworkers p --comms local
   python3 test_persistent_prox_slide.py --nworkers p --comms tcp

The number gens will be 4.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

import sys
import numpy as np
import numpy.linalg as la
import scipy.sparse as spp

from libensemble.libE import libE
from libensemble.gen_funcs.persistent_prox_slide import opt_slide as gen_f
from libensemble.alloc_funcs.start_persistent_consensus import start_consensus_persistent_gens as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.tools.consensus_subroutines import get_k_reach_chain_matrix, readin_csv, gm_opt, svm_opt

from libensemble.sim_funcs.geomedian import geomedian_eval
from libensemble.sim_funcs.svm import svm_eval

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")
if nworkers < 5:
    sys.exit('This tests requires at least 5 workers (6 MPI processes)...')

num_gens = 4
A = spp.diags([2, 3, 3, 2]) - get_k_reach_chain_matrix(num_gens, 2)
lam_max = np.amax((la.eig(A.todense())[0]).real)

eps = 1e-1

# 0: geometric median, 1: SVM prob_id = 1
for prob_id in range(1,2):
    persis_info = {}
    persis_info['print_progress'] = 0
    persis_info['A'] = A
    
    persis_info = add_unique_random_streams(persis_info, nworkers + 1)
    persis_info['gen_params'] = {}
    exit_criteria = {'elapsed_wallclock_time': 300}
    
    libE_specs['safe_mode'] = False

    if prob_id == 0:
        sim_f = geomedian_eval
        m, n = 10, 20
        prob_name = 'Geometric median'
        M = num_gens/(m**2)
        N_const = 4
        err_const = 1e2

        B = np.array([np.random.normal(loc=10, scale=1.0, size=n) for i in range(m)])
        persis_info['sim_params'] = {'B': B}
        fstar = gm_opt(np.reshape(B, newshape=(-1,)), m)

        def df(x, i):
            b_i = B[i]
            z = x-b_i
            return (1/m)*z/la.norm(z)

        def f(x, i):
            return (1/m)*la.norm(x-b_i)

        # Setting @f_i_eval and @df_i_eval tells to gen to compute gradients locally
        persis_info['gen_params'] = {'f_i_eval': f, 'df_i_eval': df}

    if prob_id == 1:
        sim_f = svm_eval
        m, n = 50, 15
        prob_name = 'SVM with l1 regularization'
        err_const = 1e1
        N_const = 1
        b, X = readin_csv('wdbc.data')
        X = X.T
        c = 0.1

        # reduce problem size
        b = b[:m]
        X = X[:n, :m]
        M = c*((m)**0.5)

        persis_info['sim_params'] = {'X': X, 'b': b, 'c': c, 'reg': 'l1'}
        fstar = svm_opt(X, b, c, reg='l1')

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
                 'user': {'lb': -np.zeros(n),
                          'ub': np.zeros(n)}
                 }

    alloc_specs = {'alloc_f': alloc_f,
                   'out': [],
                   'user': {'m': m, 'num_gens': num_gens},
                   }

    # Include @f_i_eval and @df_i_eval if we want to compute gradient in gen
    persis_info['gen_params'].update({'M': M,
                                      'R': 10**2,
                                      'nu': 1,
                                      'eps': eps,
                                      'D': 2*n,
                                      'N_const': N_const,
                                      'lam_max': lam_max})

    if is_manager:
        print('=== Optimizing {} ==='.format(prob_name), flush=True)

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                alloc_specs, libE_specs)

    if is_manager:
        print('=== End algorithm ===', flush=True)

    if is_manager:
        # check we completed
        assert flag == 0

        # check we have a Laplacian matrix
        assert la.norm(A.dot(np.zeros(A.shape[1]))) < 1e-15, 'Not a Laplacian matrix'

        # compile sum of {f_i} and {x}, and check their values are bounded by O(eps)
        eval_H = H[H['eval_pt']]

        gen_ids = np.unique(eval_H['gen_worker'])
        assert len(gen_ids) == num_gens, 'Gen did not submit any function eval requests'

        x = np.empty(n*num_gens, dtype=float)
        F = 0

        for i, gen_id in enumerate(gen_ids):
            last_eval_idx = np.where(eval_H['gen_worker'] == gen_id)[0][-1]

            f_i = eval_H[last_eval_idx]['f_i']
            x_i = eval_H[last_eval_idx]['x']

            F += f_i
            x[i*n:(i+1)*n] = x_i

        A_kron_I = spp.kron(A, spp.eye(n))
        consensus_val = np.dot(x, A_kron_I.dot(x))

        assert F-fstar < err_const*eps, 'Error of {:.4e}, expected {:.4e} (assuming f*={:.4e})'.format(
            F-fstar, err_const*eps, fstar)
        assert consensus_val < eps, 'Consensus score of {:.4e}, expected {:.4e}\nx={}'.format(
            consensus_val, eps, x)
