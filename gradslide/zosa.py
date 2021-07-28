import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from helper import *
import sys

def zoSA(settings, x0=None, simple_l1=False):
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

    # print('x={} '.format(x_post))
    print('gap={}'.format(f_eval(x) - fstar))
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
            # grad_x_k = 2*R*A.dot(pre_x) + df_smooth(pre_x)
            return grad_x_k

        if not simple_l1:
            print('{}/{}: {} iters'.format(k, N, T_k))
            [x,x_apprx] = prox_slide(df, h, l_f, h_k, dh_k, x_prev, b_k, T_k, k )

            x_post = (1.0-g_k)*(x_post) + g_k*(x_apprx)
        else:
            # solve non-smooth argmin when simple, e.g. l1
            print('{}/{}'.format(k, N))
            gradf = df_smooth(pre_x) + dh_k()
            u_star = (b_k*x_prev) - gradf
            is_nonzero_idxs = (np.abs(u_star) > 1).astype(int)
            x = np.multiply(is_nonzero_idxs, u_star - np.sign(u_star))
            x = x/b_k

            x_post = (1.0-g_k)*(x_post) + g_k*(x)

        # print('x={} '.format(x_post))
        print('gap={}'.format(f_eval(x_post) - fstar))
        print('consensus={}\n'.format(consensus(x_post)/R))
        # print('||gradf||_2={:.4f}'.format(la.norm(df(x) + dh_k() + np.sign(x), ord=2)))

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
        # projection (ignore for now)
        # u_norm = la.norm(u_next, ord=2)
        # Lim = 24
        # if u_norm > Lim:
        #     u_next *= Lim/u_norm

        u_tilde = (1.0-theta_t) * u_tilde + theta_t * u_next

        u = u_next

    return [u, u_tilde]

if len(sys.argv) < 7:
    print('python zosa.py --graph {1,2,3} --prob {1,2,3,4} --start {1,2,3} [--simple]')
    print('\t{1,2,3}={chain,random,complete}, {1,2,3}={Geomed, SVM, Lin+l1, Log+l1}, {1,2,3}={random,fixed,zero}')
    exit(0)

seed_num = 0
if len(sys.argv)>=9 and sys.argv[7]=='--seed':
    seed_num = int(sys.argv[8])

[graph_mode,prob_mode, x_0_mode] = [int(i) for i in sys.argv[2:7:2]]
simple_l1 = False

######## SETUP #################################
np.random.seed(seed_num)
n = 100

# Geometric median
if prob_mode == 1:
    n = 10
    m = 100
    b = np.array([np.random.normal(loc=0, scale=10.0, size=n) for i in range(m)])
    b = np.reshape(b, newshape=(-1,))

    def df(x): return df_gm_long(x,b,m)
    def f_eval(x): return f_gm(x,b,m)

    xstar = gm_opt(b,m)
    fstar = f_gm_comb(xstar,b,m)
    xstar = np.kron(xstar, np.ones(m))

    M = 1

# SVM 
elif prob_mode == 2:
    b, X = readin_csv('wdbc.data')
    X = X.T
    reg = 'l1'

    d,m = X.shape
    n = d

    def df(theta): return df_svm(theta, X, b, reg)
    def f_eval(theta): return f_svm_long(theta, X, b, reg)

    c = 0.1
    M = c*((n*m)**0.5)

    xstar = svm_opt(X, b, reg)
    fstar = f_svm(xstar, X, b, reg)

# Linear Reg with l1
elif prob_mode == 3:
    reg = 'l1'
    m = n
    d = 10
    n = d

    mean = 10*np.random.random()-5
    var  = 10.0
    X = np.array([np.random.normal(loc=mean, scale=var, size=d) for _ in range(m)]).T
    y = np.dot(X.T, np.ones(d)) + np.cos(np.dot(X.T, np.ones(d))) + np.random.normal(loc=0, scale=0.25, size=m)
    u = np.ones(d)
    y = np.dot(X.T,u)

    def df_smooth(theta): return df_regls(theta, X, y, reg=None)
    # def df(theta): return np.sign(theta)
    def df(theta): return df_regls(theta, X, y, reg=reg)
    def f_eval(theta): return f_regls_long(theta, X, y, reg=reg)

    c = 0.1
    M = c*((n*m)**0.5)
    # eigenvalue approach
    eig_max = np.amax(la.eig(np.dot(X,X.T))[0].real)
    L = eig_max/m

    # each element approach
    # X_norms = la.norm(X, ord=2, axis=0)**2
    # L = (2/m)*(np.amax(X_norms)+c)

    xstar = regls_opt(X, y, reg=reg)
    fstar = f_regls(xstar, X, y, reg=reg)
    xstar = np.kron(np.ones(m), xstar)

    # simple_l1 = True

# Logistic Reg with l1
elif prob_mode == 4:
    reg = 'l1'
    m = n
    d = 10
    n = d
    y = np.append(2*np.ones(m//2), np.zeros(m-m//2))-1
    X = np.array([np.random.normal(loc=y[i]*np.ones(d), scale=1.0, size=d) for i in range(m)]).T

    # TODO: Why can't we solve inner loop 
    def df_smooth(theta): return df_log(theta, X, y, reg=None)
    # def df(theta): return np.sign(theta)
    def df(theta): return df_log(theta, X, y, reg=reg)
    def f_eval(theta): return f_log_long(theta, X, y, reg=reg)

    c = 0.1
    M = c*((n*m)**0.5)
    # eigenvalue approach
    XXT_sum = np.outer(X[:,0], X[:,0])
    for i in range(1,m):
        XXT_sum += np.outer(X[:,i],X[:,i])
    eig_max = np.amax(la.eig(XXT_sum)[0].real)
    L = eig_max/m

    xstar = log_opt(X, y, reg=reg)
    fstar = f_log(xstar, X, y, reg=reg)

    # simple_l1 = False

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
    p = 0.15
    A = get_er_graph(m, p, seed=0)
elif graph_mode==3: 
    k = m-1
    A = k*spp.eye(m) - get_k_reach_chain_matrix(m,k)
else:
    print('Invalid graph {}'.format(graph_mode))
    exit(0)
A_norm = la.norm(A.todense(), ord=2)
assert la.norm(A.dot(np.ones(A.shape[1]))) < 1e-15
lam_max = np.amax((la.eig(A.todense())[0]).real)
A = spp.kron(A, spp.eye(n))

D = 2*n # 3/4 * 2 * n**0.5
R = 10**2
eps = 0.1
const = 10
if prob_mode >= 3:
    L += 2*R*lam_max 
else:
    L = 2*R*lam_max 
nu = 1
N = const * int(((L*D/(nu*eps))**0.5 + 1))

settings = { 'D': D, 'L': L, 'M': M, 'N': N, 'R': R, 'm': m, 'n': n, 'A': A, 'fstar': fstar, 'f': f_eval }
####### PARMETERS #########################

xstar = zoSA(settings, x, simple_l1)
