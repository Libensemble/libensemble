import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from helper import *

"""
Change
- Size of problem: m, n, k (Line 174)
- Initial starting point x and xstar (Line 190)
- df() as well as gradient Lipschitz constant L (Line 185)
- fstar (optimal solution, if known) (Line 209)
- f() for gap computation (Line 211)
"""

def primaldual(x_0, df, settings):
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
    print('x={}'.format(x_0))
    print('gap={}'.format(f_eval(x_0) - fstar))
    print('consensus={}\n'.format(cons))

    for k in range(1,N+1):
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
                    'Lap': Lap
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
        print('[{}/{}]\nx={}'.format(k, N, _x))
        print('gap={}'.format(f_eval(_x) - fstar))
        print('numcomms={}'.format(T_k))
        print('consensus={}'.format(cons))
        print('')

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

        # print(' >> {}'.format(x_curr))

    x_k   = x_curr
    x_k_1 = x_prev
    z_k   = z_t
    x_hk  = xsum/T_k

    return [x_k, x_k_1, z_k, x_hk]

####### PARAMETERS ####################
n = 100
m = n-1
k = m-1
# k = 1
####### PARAMETERS ####################

####### FUNCTIONS ####################
b = np.ones(m*n)
def df(x): return df_ar(x)
def V(x,u): return 0.5*la.norm(x-u, ord=2)**2
####### FUNCTIONS ####################

####### PARAMETERS ####################
# random guess
np.random.seed(0)
x_0 = 2*np.random.random(m*n)-1

if k == m-1:
    A = k*spp.eye(m) - get_k_reach_chain_matrix(m,k)
else:
    assert k==1
    A = spp.diags(np.append(1, np.append(2*np.ones(m-2), 1))) - get_k_reach_chain_matrix(m,k)
A_norm = la.norm(A.todense(), ord=2)
assert la.norm(A.dot(np.ones(A.shape[1]))) < 1e-15
A = spp.kron(A, spp.eye(n))

x_sol = np.ones(len(x_0), dtype=float)
mu = 0
L = 1
# change constant in front ...
Vx_0x = V(x_0, x_sol)
R = 1 / (4 * (Vx_0x)**0.5)
eps = 1e-3
N = int(4 * (L*Vx_0x/eps)**0.5 + 1)

fstar = 0
settings = { 'mu': mu, 'R': R, 'L': L, 'Lap': A, 'N': N, 'A_norm': A_norm, 'fstar': fstar, 'f': f_ar_long }
####### PARAMETERS ####################

xstar = primaldual(x_0, df, settings)
print('Sol:', xstar)
