import numpy as np
import numpy.linalg as la
import scipy.sparse as spp

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

def opt_slide(H, persis_info, gen_specs, libE_info):
    """ Gradient sliding. Coordinates with alloc to do local and distributed 
        (i.e., gradient of consensus step) calculations.
    """
    # Send batches until manager sends stop tag
    tag = None
    gen_id = persis_info['worker_num']
    # each gen has unique interal id
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    f_i_idxs = persis_info['f_i_idxs']
    n = len(ub)

    # start with random x0
    x0 = persis_info['rand_stream'].uniform(low=lb, high=ub, size=(n,)) 
    x_k = x0
    post_x_k = x0

    print_progress = True
    if print_progress:
        print('[{}]: x={}'.format(gen_id, x_k), flush=True)

    # define variables
    M       = persis_info['params']['M']
    R       = persis_info['params']['R']
    nu      = persis_info['params']['nu']
    eps     = persis_info['params']['eps']
    D       = persis_info['params']['D']
    N_const = persis_info['params']['N_const']
    lam_max = persis_info['params']['lam_max']

    L = 2*R*lam_max
    N = N_const * int( ((L * D)/(nu * eps) )**0.5 + 1 )

    # print('D={}, L={}, M={}, N={}'.format(D, L, M, N), flush=True)

    for k in range(1,N+1):
        b_k = 2.0*L/(nu * k)
        g_k = 2.0/(k+1)
        T_k = int( (N*((M*k)**2)) / (D*(L**2)) + 1 )

        pre_x_k = (1.0 - g_k) * post_x_k + (g_k * x_k)

        H_o = np.zeros(1, dtype=gen_specs['out'])
        H_o['x'][0] = pre_x_k
        H_o['consensus_pt'][0] = True

        if print_progress and gen_id==1:
            print('[{}/{}]: {} inner iters'.format(k, N, T_k), flush=True)

        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        # compute consensus step, i.e., take linear combination of neighbors
        neighbors_pre_x_k = calc_in['x']
        num_neighbors = len(neighbors_pre_x_k)
        gradg = 2*R * ( num_neighbors * pre_x_k - np.sum(neighbors_pre_x_k, axis=0) )
        _x_k = x_k.copy()
        x_k, x2_k = PS(_x_k, gradg, b_k, T_k, f_i_idxs, gen_specs, libE_info, pre_x_k, persis_info['worker_num'])

        # If sent back nothing, must have crashed
        if x_k is None:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        post_x_k = (1.0-g_k) * post_x_k + (g_k * x2_k)
        if print_progress:
            print('[{}]: x={}'.format(gen_id, post_x_k), flush=True)

    if print_progress:
        print('[{}]: x={}'.format(gen_id, post_x_k))
        print_final_score(post_x_k, f_i_idxs, gen_specs, libE_info)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def PS(x, gradg, beta, T, f_i_idxs, gen_specs, libE_info, pre_x_k, wid):
    """ Prox-sliding procedure (see https://arxiv.org/pdf/1911.10645) with
        entropy as the distance generating function for Bregman divergence.
    """
    u = x
    u2 = x
    n = len(x)
    l = len(f_i_idxs)

    for t in range(1,T+1):

        # request grad_u f(u_{t-1})
        H_o = np.zeros(l, dtype=gen_specs['out'])
        H_o['x'][:] = u
        H_o['consensus_pt'][:] = False
        H_o['obj_component'][:] = f_i_idxs
        H_o['get_grad'][:] = True
        # unfold into 1d array
        H_o = np.reshape(H_o, newshape=(-1,))      

        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, None

        p_t = t/2.0
        theta_t = 2.0*(t+1)/(t * (t+3))

        # add gradients since we have sum of convex functions

        gradf_is = calc_in['gradf_i']
        gradf    = np.sum(gradf_is, axis=0)

        u_next = get_l2_argmin(x, u, gradf, gradg, beta, p_t)
        u = u_next
        u2 = (1-theta_t) * u2 + (theta_t * u)

    return u, u2

def get_l2_argmin(x, u_prev, gradf, gradg, beta, p_t):
    u_next = (beta * x) + (beta * p_t * u_prev) - gradf - gradg
    u_next = u_next / (beta * (1.0+p_t))
    return u_next
    
    """
    # X={ x : |x| <= 2\sqrt{n} }
    n = len(x)
    u_norm = la.norm(u_next, ord=2)
    B = 2*(n**0.5)
    if u_norm > B:
        u_next *= (B/u_norm)
    return u_next
    """

def get_entropy_argmin(x, u, gradf, gradg, beta, p_t):
    n = len(x) 

    const = 1.0/(1+p_t) * (np.log(x) + (1+p_t) * np.ones(n, dtype=float)) \
            - 2/(beta * (1+p_t))*gradg

    dyn = p_t/(1+p_t) * np.log(u) - 1.0/( beta * (1+p_t) ) * gradf

    u_next = np.exp( const + dyn )

    return u_next

def print_final_score(x, f_i_idxs, gen_specs, libE_info):

   # evaluate { f_i(x) } first
   l = len(f_i_idxs)
   H_o = np.zeros(l, dtype=gen_specs['out'])
   H_o['x'][:] = x
   H_o['consensus_pt'][:] = False
   H_o['obj_component'][:] = f_i_idxs
   H_o['get_grad'][:] = False
   H_o = np.reshape(H_o, newshape=(-1,))      

   tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
   f_is = calc_in['f_i']
   F_i  = np.sum(f_is)

   # get alloc to print sum of f_i
   H_o = np.zeros(1, dtype=gen_specs['out'])
   H_o['x'][0] = x
   H_o['f_i'][0] = F_i
   H_o['eval_pt'][0] = True
   H_o['consensus_pt'][0] = True

   tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

