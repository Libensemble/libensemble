import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from helper import *

"""
Change
- Size of problem: m, n, k (Line 147)
- Initial starting point x and xstar (Line 166)
- df() as well as gradient Lipschitz constant L (Line 179)
- fstar (optimal solution, if known) (Line 168)
- f() for gap computation (Line 170)
"""

def zoSA(settings, x0=None):
    """ Zeroth-order sliding algorithm 

    Returns
    -------
    barx_N : np.ndarray
        solution
    """
    D = settings['D']
    L = settings['L']
    M = settings['M']
    N = settings['N']
    R = settings['R']
    m = settings['m']
    n = settings['n']
    A = settings['A']
    f_eval = settings['f']
    fstar = settings['fstar']

    np.random.seed(1)
    if x0 is None:
        x0 = 2*np.random.random(m*n)-1
    x = x0
    x_post = x0
    
    np.random.seed(0)
    b = np.random.random(m*n)
    # b = np.ones(m*n)

    def h(x):
        return R*np.dot(x, A.dot(x))

    def l_f(u_0,u):
        g_u = R*np.dot(u_0, A.dot(u_0))
        grad_u = 2*R*A.dot(u_0)
        return g_u + np.dot(grad_u, u-u_0)

    def consensus(x):
        score1 = R * np.dot(x, A.dot(x))
        return score1

    print('x={} '.format(x_post))
    print('gap={}'.format(f_eval(x_post[:n]) - fstar))
    print('consensus={}\n'.format(consensus(x_post)/R))

    for k in range(1,N+1):
        x_prev = x
        b_k = 2*L/(nu*k)
        g_k = 2/(k+1)
        T_k = int( ((M**2)*N*(k**2))/(D*(L**2)) + 1)

        pre_x = (1.0-g_k)*(x_post) + g_k*(x_prev)

        # linear form and its gradient
        def h_k(u):
            g_x_k = R*np.dot(pre_x, A.dot(pre_x))
            grad_x_k = 2*R*A.dot(pre_x)
            return g_x_k + np.dot(grad_x_k, u-pre_x)

        def dh_k():
            grad_x_k = 2*R*A.dot(pre_x)
            return grad_x_k

        print('{}/{}: {} iters'.format(k, N, T_k))
        [x,x_apprx] = prox_slide(df, h, l_f, h_k, dh_k, x_prev, b_k, T_k, k )

        x_post = (1.0-g_k)*(x_post) + g_k*(x_apprx)

        print('x={} '.format(x_post))
        print('gap={}'.format(f_ar(x_post[:n]) - fstar))
        print('consensus={}\n'.format(consensus(x_post)/R))

        if np.any(np.isnan(x_post)):
            print('\nExitted due to infinity\n')
            break

    return x_post
    
def prox_slide(df, H, l_h, h, dh, x, beta, T, k):
    """ Proximal sliding

    Parmaters
    ---------
    h :: func : np.ndarray -> np.ndarray
        linear grad approximation

    x : np.ndarray
        initial point

    beta : float
        positive scalar

    T : int
        number of iterations

    MISSING: {p_t}, {theta_t}

    Returns
    -------
    x : np.ndarray
        first solution

    x_apprx : np.ndarray
        second solution
    """
    u = x
    u_tilde = x
    n = len(u)

    P_t = 1
    A = 0

    for t in range(1,T+1):
        p_t = t/2.0
        theta_t = 2*(t+1)/(t*(t+3))

        # compute argmin
        u_prev = u
        u = (beta * x) + (beta * p_t * u_prev) - dh() - df(u_prev)
        u_next = u/(beta + beta*p_t)
        # projection
        u_norm = la.norm(u_next, ord=2)
        Lim = 24
        if u_norm > Lim:
            u_next *= Lim/u_norm

        u_tilde = (1.0-theta_t) * u_tilde + theta_t * u_next

        u = u_next

    return [u, u_tilde]


####### PARMETERS #########################
n = 100
m = n//2
k = 1
A = get_k_reach_chain_matrix(m,k)
A = -A + spp.diags([np.append(1, np.append(2*np.ones(m-2), 1))], [0])
assert la.norm(A.dot(np.ones(A.shape[0]))) < 1e-15
A = spp.kron(A, spp.eye(n))
lam_max = np.amax(la.eig(A.todense())[0]).real

M = 1
D = 2*n # 3/4 * 2 * n**0.5
R = 10**2
eps = 0.1
const = 7
L = 2*R*lam_max 
nu = 1
N = const * int(((L*D/(nu*eps))**0.5 + 1))

x = get_square_x0()
fstar = 0

settings = { 'D': D, 'L': L, 'M': M, 'N': N, 'R': R, 'm': m, 'n': n, 'A': A, 'fstar': fstar, 'f': f_r_long }
####### PARMETERS #########################

####### FUNCTIONS #########################
np.random.seed(0)
d = n
X = np.random.normal(loc=0, scale=1, size=(d,m))
ones_d = np.ones(d)
# y = np.dot(X.T, ones_d) + np.cos(np.dot(X.T, ones_d)) + np.random.normal(loc=0, scale=0.25, size=m)
y = np.dot(X.T, ones_d) # + np.random.normal(loc=0, scale=0.25, size=m)
def df(x): return df_regls(x, X, y)
def V(u_0, u): return 1/2 * la.norm(u_0-u, ord=2)**2
def _g(u): return R*np.dot(u, A.dot(u))
def _dg(u): return 2*R*A.dot(u)
####### FUNCTIONS #########################

xstar = zoSA(settings, x)
