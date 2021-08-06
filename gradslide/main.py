import numpy as np
import numpy.linalg as la
from pycute_interface import Blackbox
from helper import *

"""
TODO: Why are the algorithms converging much more poorly than I anticipated?
"""

m = 20
n = 100

GAP = 1e-3
CON = 1e-6
MAXITER = 500000

def nest(seed_num):
    bbox = Blackbox()
    bbox.setup_new_prob(seed_num)
    bbox.set_scale()

    [fstar, xstar] = optimize_blackbox(bbox)
    fstar = fstar/bbox.get_scale()

    x = np.zeros(n)
    L = 1

    def df(theta): return bbox.df(theta)
    def f_eval(theta): return bbox.f(theta)/bbox.get_scale()

    beta = L
    beta_inv = 1.0/beta
    eps = 1e-6
    N = int((2*L/eps * la.norm(xstar-x,ord=2)**2)**0.5 + 1)
    N = min(N, MAXITER)
    lam_prev = 0
    lam_curr = 1
    y_curr = x.copy()

    gap0 = f_eval(x) - fstar

    for k in range(1,N+1):
        gamma = (1-lam_prev)/lam_curr
        lam_prev = lam_curr
        lam_curr = (1 + (1+4*lam_prev)**0.5)/2

        y_next = x - beta_inv*df(x)
        x = (1-gamma)*y_next + gamma*y_curr
        y_curr = y_next

        gap = f_eval(y_curr) - fstar

        if gap/gap0 < GAP:
            return k

    return -1

def nagent(Lap, W, rho, seed_num, step_scale=1):
    """ Setups up problem for nagent and then runs it """
    L = 1
    eta = 100/L * min(1/6, (1-rho**2)**2/(4*rho**2*(3+4*rho**2)))
    N = MAXITER

    # setup problem
    bbox = Blackbox()
    bbox.setup_new_prob(seed_num)
    bbox.set_scale()

    def df(theta): return bbox.df_long(theta)
    def f_eval(theta):
        return bbox.f_long(theta)/bbox.get_scale()

    [fstar, xstar] = optimize_blackbox(bbox)
    fstar = fstar/bbox.get_scale()
    xstar = np.kron(np.ones(m), xstar)

    settings = { 'm': m, 'W': W, 'L': L, 'N': N, 'eta': eta * step_scale, 'Lap': Lap, 'fstar': fstar, 'df': df, 'f': f_eval, }

    return nagent_alg(settings)

def nagent_alg(settings):
    """ Returns number of iterations to reduce absolute error
        by 1e-3 and have consensus reach < 1e-6.
    """
    W = settings['W']
    L = settings['L']
    N = settings['N']
    eta = settings['eta']
    Lap = settings['Lap']
    fstar = settings['fstar']
    df = settings['df']
    f_eval = settings['f']

    x = np.zeros(n*m)
    gap0 = f_eval(x) - fstar
    s = np.zeros(len(x), dtype=float)
    g_prv = s.copy()

    for k in range(1,N+1):
        g = df(x)

        # since we allow large step sizes, see if we take wild steps
        if np.any(np.isinf(g)):
            return -1

        s = W.dot(s + g - g_prv)
        x = W.dot(x-eta*s)
        g_prv = g

        gap = f_eval(x) - fstar
        con = np.dot(x, Lap.dot(x))/np.dot(x,x)

        if gap/gap0 < GAP and con < CON:
            return k

    return -1

def primaldual_alg(x_0, df, settings,maxiter=-1):

    # immutable variables
    mu              = settings['mu']
    R               = settings['R']
    L               = settings['L']
    Lap             = settings['Lap']
    N               = settings['N']
    A_norm          = settings['A_norm']
    fstar           = settings['fstar']
    f_eval          = settings['f']

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

    gap0 = f_eval(x_0) - fstar
    total_inner_iters = 0

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

        settings = {'T_k': T_k, 'b_k': k, 'p_k': 2*L/k, 'mu': mu, 'L': L, 'R': R, 'k': k, 'prev_b_k': prev_b_k, 'prev_T_k': prev_T_k, 'Lap': Lap, 'f_eval': f_eval, 'fstar': fstar, 'curr_x_sum': weighted_x_hk_sum, 'curr_b_k_sum': b_k_sum, 'gap0': gap0}

        [x_k, x_k_1, z_k, x_hk, niters] = primaldual_slide(y_k, prev_x_k, prev_penult_k, prev_z_k, settings)

        # we required all iterations to converge
        if niters == -1:
            total_inner_iters += T_k
        else:
            total_inner_iters += niters
            return [k, total_inner_iters]
        if maxiter > 0 and total_inner_iters > maxiter:
            return [-1,-1]

        prevprev_x_k = prev_x_k
        prev_x_k      = x_k
        prev_x_hk     = x_hk
        prev_penult_k = x_k_1 # penultimate x_k^{(i)}
        prev_z_k      = z_k
        prev_b_k      = b_k
        prev_T_k      = T_k
        prev_x_uk = x_uk

        weighted_x_hk_sum += b_k * x_hk
        b_k_sum += b_k

        _x = 1.0/b_k_sum * weighted_x_hk_sum
        con = np.dot(_x, Lap.dot(_x))/np.dot(_x,_x)

    return [-1,-1]

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

    gap0   = settings['gap0']
    f_eval = settings['f_eval']
    fstar  = settings['fstar']

    x_k_1 = x_curr.copy()
    xsum = np.zeros(len(x_curr), dtype=float)

    for t in range(1,T_k+1):
        # define per-iter params
        eta_t = (p_k + mu)*(t-1) + p_k*T_k
        q_t   = L*T_k/(2*b_k*R**2)
        if k >= 2 and t == 1: a_t = prev_b_k*T_k/(b_k*prev_T_k)
        else: a_t = 1

        u_t = x_curr + a_t * (x_curr - x_prev)
        z_t = z_t + (1.0/q_t) * Lap.dot(u_t)
        x_next = (eta_t*x_curr) + (p_k*x_k_1) - (y_k + Lap.dot(z_t))
        x_next /= (eta_t + p_k)

        x_prev = x_curr
        x_curr = x_next

        xsum += x_curr
        curr_x_in = xsum/t
        curr_x_est = (settings['curr_x_sum'] + b_k*curr_x_in)/(settings['curr_b_k_sum'] + b_k)

        gap  = f_eval(curr_x_est) - fstar
        con = np.dot(curr_x_est, Lap.dot(curr_x_est))/np.dot(curr_x_est, curr_x_est)

        # reached convergence
        if gap/gap0 < GAP and con < CON:
            return [x_curr, x_prev, z_t, xsum, t]

    x_k   = x_curr
    x_k_1 = x_prev
    z_k   = z_t
    x_hk  = xsum/T_k

    return [x_k, x_k_1, z_k, x_hk, -1]

def pds(A, A_norm, seed_num):
    bbox = Blackbox()
    bbox.setup_new_prob(seed_num)
    bbox.set_scale()
    L = 1

    def df(theta): return bbox.df_long(theta)
    def f_eval(theta): return bbox.f_long(theta)/bbox.get_scale()

    [fstar, xstar] = optimize_blackbox(bbox)
    fstar = fstar/bbox.get_scale()
    xstar = np.kron(np.ones(m), xstar)
    x = np.zeros(n*m)

    mu = 0
    Vx_0x = V(x, xstar)
    R = 1 / (4 * (Vx_0x)**0.5)
    eps = 1e-3
    N = int(4 * (L*Vx_0x/eps)**0.5 + 1)

    settings = { 'mu': mu, 'R': R, 'L': L, 'Lap': A, 'N': N, 'A_norm': A_norm, 'fstar': fstar, 'f': f_eval }

    return primaldual_alg(x, df, settings, maxiter=MAXITER)

def main():
    num_trials = 100

    As = []
    Ws = []
    rhos = []
    A_norms = []

    # generate communication graphs
    print('Generating graphs apriori')
    np.random.seed(0)
    k = 1
    A1 = spp.diags(np.append(1, np.append(2*np.ones(m-2), 1))) - get_k_reach_chain_matrix(m,k)
    W1 = get_doubly_stochastic(A1)
    rho1 = la.norm(W1 - n**-1*np.ones((m,m)), ord=2)
    A_norm1 = la.norm(A1.toarray(), ord=2)
    As.append(spp.kron(A1, spp.eye(n)))
    Ws.append(spp.kron(W1, spp.eye(n)))
    rhos.append(rho1)
    A_norms.append(A_norm1)

    p = 0.227
    A2 = get_er_graph(m, p, seed=0)
    W2 = get_doubly_stochastic(A2)
    rho2 = la.norm(W2 - n**-1*np.ones((m,m)), ord=2)
    A_norm2 = la.norm(A2.toarray(), ord=2)
    As.append(spp.kron(A2, spp.eye(n)))
    Ws.append(spp.kron(W2, spp.eye(n)))
    rhos.append(rho2)
    A_norms.append(A_norm2)

    k = m-1
    A3 = k*spp.eye(m) - get_k_reach_chain_matrix(m,k)
    W3 = get_doubly_stochastic(A3)
    A_norm3 = la.norm(A3.toarray(), ord=2)
    rho3 = la.norm(W3 - n**-1*np.ones((m,m)), ord=2)
    As.append(spp.kron(A3, spp.eye(n)))
    Ws.append(spp.kron(W3, spp.eye(n)))
    rhos.append(rho3)
    A_norms.append(A_norm3)
    print('Finished generating graphs\n')

    print('Centralized GD:')
    for s in range(1,num_trials+1):

        # centralized
        niters = nest(s)
        if s < num_trials:
            print('{},'.format(niters), end='', flush=True)
        else:
            print(niters)

    for p in range(0,4):
        step_scale = int(10**p)

        for g in range(3):
            print('\nDecentralized GD({},{})'.format(step_scale, g))

            for s in range(1,num_trials+1):
                A = As[g]
                W = Ws[g]
                rho = rhos[g]
                niters = nagent(A, W, rho, s, step_scale=step_scale)
                if s < num_trials:
                    print('{},'.format(niters), end='', flush=True)
                else:
                    print(niters)

    for g in range(3):
        print('\nPDS({})'.format(g))
        for s in range(1,num_trials+1):
            A = As[g]
            A_norm = A_norms[g]
            [n_out, n_in] = pds(A, A_norm, s)
            if s < num_trials:
                print('{},{},'.format(n_out, n_in), end='', flush=True)
            else:
                print('{},{}'.format(n_out,n_in))

if __name__ == '__main__':
    main()
