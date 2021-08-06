import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from helper import *
import sys
from pycute_interface import Blackbox

def primaldual(x_0, df, settings,maxiter=-1):
    """ Primal-dual sliding algorithm (outer loop)

    Returns
    -------
    barx_N : np.ndarray
        solution
    """

    # immutable variables
    mu              = settings['mu']
    R               = settings['R']
    L               = settings['L']
    Lap             = settings['Lap']
    num_outer_iters = settings['N']
    A_norm          = settings['A_norm']
    fstar           = settings['fstar']
    f_eval          = settings['f']

    # ===== NOTATION =====
    # x_hk == \hat{x}_k
    # x_tk == \tilde{x}_k
    # x_uk == \underline{x}_k
    # prev_x_k == x_{k-1}
    # prevprev_x_k = x_{k-2}
    # ====================

    # Initialize values
    prev_x_k     = x_0.copy()
    prev_x_uk    = x_0.copy()
    prev_x_hk    = x_0.copy()
    prev_z_k     = np.zeros(len(x_0), dtype=float)
    prevprev_x_k = x_0.copy()
    prev_penult_k= x_0.copy()

    prev_b_k = 0
    prev_T_k = 0

    weighted_x_hk_sum = np.zeros(len(x_0), dtype=float)
    b_k_sum = 0

    cons = np.dot(x_0, Lap.dot(x_0))
    print('[0/N]')
    # print('x={}'.format(x_0))
    # print('g={}'.format(df(x_0)))
    print('gap={:.3e}'.format(f_eval(x_0) - fstar))
    print('consensus={:.3e}\n'.format(cons))

    total_inner_iters = 0

    for k in range(1,N+1):
        print('[{}/{}]'.format(k, N))
        # define parameters
        tau_k = (k-1)/2
        lam_k = (k-1)/k
        b_k   = k
        p_k   = 2*L/k
        T_k   = int(k*R*A_norm/L + 1)

        x_tk = prev_x_k + lam_k*(prev_x_hk - prevprev_x_k)
        x_uk = (x_tk + tau_k*prev_x_uk)/(1+tau_k)

        y_k = df(x_uk)
        # print('x_uk={}'.format(x_uk))
        # print('|gradf|={:.4f}'.format(la.norm(x_uk, ord=2)))

        settings = {'T_k': T_k,
                    'b_k': k,
                    'p_k': 2*L/k,
                    'mu': mu,
                    'L': L,
                    'R': R,
                    'k': k,
                    'prev_b_k': prev_b_k,
                    'prev_T_k': prev_T_k,
                    'Lap': Lap,
                    'f_eval': f_eval,
                    'fstar': fstar,
                    'curr_x_sum': weighted_x_hk_sum,
                    'curr_b_k_sum': b_k_sum,
                    }

        [x_k, x_k_1, z_k, x_hk] = primaldual_slide(y_k,
                                                   prev_x_k,
                                                   prev_penult_k,
                                                   prev_z_k,
                                                   settings)


        # TYPO here (before had prev_prev)
        prevprev_x_k = prev_x_k
        prev_x_k      = x_k
        prev_x_hk     = x_hk
        prev_penult_k = x_k_1 # penultimate x_k^{(i)}
        # FORGOT THIS LINE
        prev_z_k      = z_k
        prev_b_k      = b_k
        prev_T_k      = T_k
        # NEW (this was the issue...)
        prev_x_uk = x_uk

        weighted_x_hk_sum += b_k * x_hk
        b_k_sum += b_k

        _x = 1.0/b_k_sum * weighted_x_hk_sum
        cons = np.dot(_x, Lap.dot(_x))
        # print('x={}'.format(_x))
        print('|g|={:3e}'.format(la.norm(df(_x),ord=2)))
        print('gap={:.3e}'.format(f_eval(_x) - fstar))
        print('numcomms={}'.format(T_k))
        print('consensus={:.3e}'.format(cons))
        print('')

        total_inner_iters += T_k
        if maxiter > 0 and total_inner_iters > maxiter:
            print('--- EXITTED DUE TO MAXITER ---')
            break

    # final solution (weighted average)
    x_star = 1.0/b_k_sum * weighted_x_hk_sum

    return x_star

def primaldual_slide(y_k, x_curr, x_prev, z_t, settings):

    # define params
    T_k = settings['T_k']
    b_k = settings['b_k']
    p_k = settings['p_k']
    mu  = settings['mu']
    L   = settings['L']
    R   = settings['R']
    k   = settings['k']
    Lap = settings['Lap']
    prev_b_k = settings['prev_b_k']
    prev_T_k = settings['prev_T_k']

    f_eval = settings['f_eval']
    fstar  = settings['fstar']

    x_k_1 = x_curr.copy()
    xsum = np.zeros(len(x_curr), dtype=float)

    for t in range(1,T_k+1):
        # define per-iter params
        eta_t = (p_k + mu)*(t-1) + p_k*T_k
        q_t   = L*T_k/(2*b_k*R**2)
        if k >= 2 and t == 1:
            a_t = prev_b_k*T_k/(b_k*prev_T_k)
        else:
            a_t = 1

        u_t = x_curr + a_t * (x_curr - x_prev)

        # compute first argmin
        z_t = z_t + (1.0/q_t) * Lap.dot(u_t)

        # computes second argmin
        x_next = (eta_t*x_curr) + (p_k*x_k_1) - (y_k + Lap.dot(z_t))
        x_next /= (eta_t + p_k)

        x_prev = x_curr
        x_curr = x_next

        xsum += x_curr
        # zsum += z_t

        # PRINT: Inner for loop progress
        curr_x_in = xsum/t
        curr_x_est = (settings['curr_x_sum'] + b_k*curr_x_in)/(settings['curr_b_k_sum'] + b_k)
        cons = np.dot(curr_x_est, Lap.dot(curr_x_est))
        print('in_gap={:.3e}'.format(f_eval(curr_x_est) - fstar))
        print('in_consensus={:.3e}'.format(cons))

    x_k   = x_curr
    x_k_1 = x_prev
    z_k   = z_t
    x_hk  = xsum/T_k

    return [x_k, x_k_1, z_k, x_hk]

if len(sys.argv) < 7:
    print('python pds.py --graph {1,2,3} --prob {1,2,3,4,5} --start {1,2,3}')
    print('\t{1,2,3}={chain,random,complete}, {1,2,3,4,5}={rosen1,rosen2,nest,lin_reg_l2,log_reg_l2}, {1,2,3}={random,fixed,zero}')
    exit(0)

seed_num = 0
if len(sys.argv)>=9 and sys.argv[7]=='--seed':
    seed_num = int(sys.argv[8])

[graph_mode,prob_mode, x_0_mode] = [int(i) for i in sys.argv[2:7:2]]

######## SETUP #################################
np.random.seed(seed_num)

c = 0.1
n = 100

# Rosenbrock
if prob_mode == 1:
    m = n//2
    def df(x): return df_r(x)
    def f_eval(x): return f_r_long(x)
    L = 1
    xstar = np.ones(m*n, dtype=int)
    fstar = 0

# Alternative Rosenbrock
elif prob_mode == 2:
    m = n-1
    def df(x): return df_ar(x)
    def f_eval(x): return f_ar_long(x)
    L = 1
    xstar = np.ones(m*n, dtype=int)
    fstar = 0

# Nesterov
elif prob_mode == 3:
    m = n+1
    def df(x): return df_nesterov(x)
    def f_eval(x): return f_nesterov_long(x)
    L = 1
    xstar = nesterov_opt(n)
    fstar = f_nesterov(xstar)
    xstar = np.kron(np.ones(m), xstar)

# Regularized LS
elif prob_mode == 4:
    m = n
    d = 10
    n = d
    X = np.array([np.random.normal(loc=0, scale=1.0, size=d) for _ in range(m)]).T
    assert X.shape[0] == d and X.shape[1] == m
    y = np.dot(X.T, np.ones(d)) + np.cos(np.dot(X.T, np.ones(d))) + np.random.normal(loc=0, scale=0.25, size=m)
    y = np.dot(X.T, np.ones(d))

    def df(theta): return df_regls(theta, X, y, reg='l2')
    def f_eval(theta): return f_regls_long(theta, X, y, reg='l2')

    # eigenvalue approach
    # eig_1 = la.eig(np.dot(X,X.T))[0][0]
    # L = eig_1/m + 2*c

    # each element approach
    X_norms = la.norm(X, ord=2, axis=0)**2
    L = (2/m)*(np.amax(X_norms)+c)

    xstar = regls_opt(X,y, reg='l2')
    fstar = f_regls(xstar, X, y, reg='l2')
    xstar = np.kron(np.ones(m), xstar)

# Regularized Log
elif prob_mode == 5:
    m = n
    d = 10
    n = d
    y = np.append(2*np.ones(m//2), np.zeros(m-m//2))-1
    X = np.array([np.random.normal(loc=y[i]*np.ones(d), scale=1.0, size=d) for i in range(m)]).T

    def df(theta): return df_log(theta, X, y, reg='l2')
    def f_eval(theta): return f_log_long(theta, X, y, reg='l2')

    XXT_sum = np.outer(X[:,0], X[:,0])
    for i in range(1,m):
        XXT_sum += np.outer(X[:,i],X[:,i])
    eig_max = np.amax(la.eig(XXT_sum)[0].real)
    L = eig_max/m

    reg = 'l2'
    xstar = log_opt(X, y, reg)
    fstar = f_log(xstar, X, y, reg)
    xstar = np.kron(np.ones(m), xstar)

# CUTEr
elif prob_mode == 6:
    n = 100
    m = 20

    bbox = Blackbox()
    bbox.set_scale()
    L = 1

    def df(theta): return bbox.df_long(theta)
    def f_eval(theta): return bbox.f_long(theta)/bbox.get_scale()

    [fstar, xstar] = optimize_blackbox(bbox)
    fstar = fstar/bbox.get_scale()
    xstar = np.kron(np.ones(m), xstar)

else:
    print('Invalid prob {}'.format(prob_mode))
    exit(0)

if x_0_mode == 1:
    np.random.seed(0)
    x = 2*np.random.random(m*n)-1
elif x_0_mode == 2:
    x = np.tile([-1.2,1],int(m*n//2))
elif x_0_mode == 3:
    x = np.zeros(m*n)
else:
    print('Invalid start {}'.format(x_0_mode))
    exit(0)

if graph_mode==1:
    k = 1
    A = spp.diags(np.append(1, np.append(2*np.ones(m-2), 1))) - get_k_reach_chain_matrix(m,k)
elif graph_mode==2:
    p = 0.1 if prob_mode == 2 else 0.15
    p = p if prob_mode != 6 else 0.227
    A = get_er_graph(m, p, seed=0)
elif graph_mode==3:
    k = m-1
    A = k*spp.eye(m) - get_k_reach_chain_matrix(m,k)
else:
    print('Invalid graph {}'.format(graph_mode))
    exit(0)
A_norm = la.norm(A.todense(), ord=2)
assert la.norm(A.dot(np.ones(A.shape[1]))) < 1e-15
A = spp.kron(A, spp.eye(n))

mu = 0
L = 1
Vx_0x = V(x, xstar)
R = 1 / (4 * (Vx_0x)**0.5)
eps = 1e-6
N = int(4 * (L*Vx_0x/eps)**0.5 + 1)

settings = { 'mu': mu, 'R': R, 'L': L, 'Lap': A, 'N': N, 'A_norm': A_norm, 'fstar': fstar, 'f': f_eval }
######## SETUP #################################

primaldual(x, df, settings, maxiter=5000000)
