import numpy as np
import numpy.linalg as la
import scipy.sparse as spp

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

def opt_slide(H, persis_info, gen_specs, libE_info):
    """
    This generation function always enters into persistent mode and returns
    ``gen_specs['gen_batch_size']`` uniformly sampled points the first time it
    is called. Afterwards, it returns the number of points given. This can be
    used in either a batch or asynchronous mode by adjusting the allocation
    function.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    m = gen_specs['user']['m']  
    n = len(lb)
    b = gen_specs['user']['gen_batch_size']
    
    # Send batches until manager sends stop tag
    tag = None
    ct = 0

    C = 1        # = O(1) independent of ...
    C_1 = 1      # see paper
    C_3 = n**0.5 # C_3=1 if 2-norm or sqrt(n) if 1-norm
    M = 1        # upper bound on ||f'(x)|| (i.d.k.)
    L = 1        # g is L-smooth
    Delta = 0    # bound on noise (but we have exact gradient)
    D_XV = (2*np.log(n))**0.5  # diameter of X w.r.t. Bregman divg. V
    D_X = 2
    s = D_X      # s <= D_X
    c = 1        # c = O(1) independent of n, C_1
    eps = 0.1

    p_star = np.log(n)/n  # (E[||e||*^4)**0.25 <= p_star ; =1 if 2-norm or O(sqrt( ln(n)/n )) for 1-norm
    r = 0.5 * s * C_3     # smoothing paramter
    M2 = c*(n**0.5)*C_1*M
    sigma_sq = 4 * p_star**2 * (C*n*M**2 + (n*Delta/r)**2)

    diagonals = [-np.ones(n-1), np.append(1, np.append(2*np.ones(n-2), 1)), -np.ones(n-1)]
    Wbar = spp.diags(diagonals, [-1,0,1])
    W = spp.kron(Wbar, spp.eye(m)) 
    lam_min = eps
    R_y = ( M**2/(m*lam_min) )**0.5

    N = 10      # number of iterations to reach desired accuracy, just pick some random number for now

    while 1:
        # start with random x0
        x0 = persis_info['rand_stream'].uniform(lb, ub, (n,))
        # x0 = x0/np.sum(x0)           # project to 1-simplex
        x0 = np.kron(np.ones(m), x0) # @m separate problems

        post_x = x0
        x = x0

        for k in range(1,N+1):
            b_k = 2.0*L/k
            g_k = 2.0/(k+1)
            D2  = 0.75 * D_XV**2
            T_k = int( N*(M2**2 + sigma_sq)*k**2 / (D2*L**2) )

            print("num iters:", T_k, flush=True)

            pre_x = (1-g_k) * post_x + (g_k * x)
            x, x2, ct = PS(pre_x, x, b_k, T_k, eps, R_y, W, ct, m, gen_specs, libE_info)

            if x is None:
                return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

            post_x = (1-g_k) * post_x + (g_k * x2)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def PS(x_center, x, beta, T, eps, R_y, W, ct, m, gen_specs, libE_info):
    """ Prox-sliding procedure (see https://arxiv.org/pdf/1911.10645)
    """
    u = x
    u2 = x
    assert len(x) % m == 0, "incorrect tensor product dimensions"
    n = len(x)//m

    # chain graph (use scipy later)
    Wx_center = W.dot(x_center) # one communication round
    print(">> ||Wx|| = {:.4f}".format(la.norm(Wx_center)))

    for t in range(1,T+1):
        # request gradient
        H_o = np.zeros(m, dtype=gen_specs['out'])
        H_o['x'][:] = np.reshape(u, newshape=(m,n)) 
        H_o['pt_id'][:] = ct      # unique pt id corresponding to this gen 
        H_o['obj_component'][:] = np.arange(0,m)
        ct += 1
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, None, ct

        p_t = t/2
        theta_t = 2*(t+1)/(t * (t+3))
        const = 1.0/(1+p_t) * (np.log(x) + (1+p_t) * np.ones(n*m, dtype=float)) \
                - (2*R_y**2)/(beta * (1+p_t) * eps)*Wx_center

        grad_f = calc_in['gradf_i']
        grad_f = np.reshape(grad_f, newshape=(-1,))
        assert len(grad_f) == len(u), print("len(grad_f)={}, expected {}".format(len(grad_f), len(u)))

        dyn = p_t/(1+p_t) * np.log(u) - 1.0/( beta * (1+p_t) ) * grad_f

        u_next = np.exp( const + dyn )

        # this is last line for argmin computation (returns vector in 1-simplex)
        # u = u_next/np.sum(u_next) 
        u = u_next

        u2 = (1-theta_t) * u2 + (theta_t * u)

    return u, u2, ct
