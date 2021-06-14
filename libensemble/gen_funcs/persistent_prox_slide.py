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
    ct = 0
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(ub)

    # start with random x0
    x0 = persis_info['rand_stream'].uniform(lb, ub, (n,))
    x_k = x0
    post_x_k = x0

    # immutable variables
    R_y = persis_info['R_y']
    eps = persis_info['eps']
    b_k = persis_info['beta_k']
    g_k = persis_info['gamma_k']
    f_i_idxs = persis_info['f_i_idxs']
    num_outer_iters = persis_info['N']

    for k in range(num_outer_iters):

        pre_x_k = (1.0 - g_k) * post_x_k + (g_k * x_k)

        H_o = np.zeros(1, dtype=gen_specs['out'])
        H_o['x'][0] = pre_x_k
        H_o['pt_id'][0] = ct      
        ct += 1

        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        # TODO: Error achecking
        # TODO: Not sending what we want, something like tag=1, Work=21, calc_in=None

        # mutable (i.e., changes between outer iters) variables 
        b_k = Work['persis_info']['beta_k']
        g_k = Work['persis_info']['gamma_k']
        num_inner_iters = Work['persis_info']['T_k']

        assert 'x' in calc_in, print("Alloc did not send neighboring {x_k underscore} as anticipated ...")
        neighbors_pre_x_k = calc_in['x']
        num_neighbors = len(neighbors_pre_x_k)
        # computes local grad_x { (W \otimes I)[x_1,x_2,...,x_m] }
        gradg = 2*R_y/eps * ( num_neighbors * pre_x_k - np.sum(neighbor_pre_x_k, axis=0) )

        for t in range(num_inner_iters):

            x_k, x2_k, ct = PS(x_k, gradg, b_k, T_k, eps, R_y, W, ct, f_i_idxs, gen_specs, libE_info)

            # recieved end signal 
            if x is None:
                return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

            post_x_k = (1.0-g_k) * post_x_k + (g_k * x2_k)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def PS(x, gradg, beta, T, eps, R_y, W, ct, f_i_idxs, gen_specs, libE_info):
    """ Prox-sliding procedure (see https://arxiv.org/pdf/1911.10645) with
        entropy as the distance generating function for Bregman divergence.
    """
    u = x
    u2 = x
    l = len(f_i_idxs)
    assert len(x) % l == 0, "incorrect tensor product dimensions"
    n = len(x)//l

    for t in range(1,T+1):
        # request grad_u f(u_{t-1})
        H_o = np.zeros(l, dtype=gen_specs['out'])
        H_o['x'][:] = np.reshape(u, newshape=(l,n)) 
        H_o['pt_id'][:] = ct      # unique pt id corresponding to this gen 
        H_o['obj_component'][:] = f_i_idxs
        ct += 1
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, None, ct

        assert 'gradf_i' in calc_in, print("Alloc did not send gradf_i as anticipated ...")

        p_t = t/2

        # computes argmin
        theta_t = 2*(t+1)/(t * (t+3))
        const = 1.0/(1+p_t) * (np.log(x) + (1+p_t) * np.ones(n*l, dtype=float)) \
                - (2*R_y**2)/(beta * (1+p_t) * eps)*gradg

        grad_f = calc_in['gradf_i']
        # TODO: We need to sum... right?
        grad_f = np.reshape(grad_f, newshape=(-1,))
        assert len(grad_f) == len(u), print("len(grad_f)={}, expected {}".format(len(grad_f), len(u)))

        dyn = p_t/(1+p_t) * np.log(u) - 1.0/( beta * (1+p_t) ) * grad_f

        u_next = np.exp( const + dyn )

        u = u_next
        # argmin computation over

        u2 = (1-theta_t) * u2 + (theta_t * u)

    return u, u2, ct
