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
    x0 = persis_info['rand_stream'].uniform(low=lb, high=ub, size=(n,)) 
    x_k = x0
    post_x_k = x0

    # immutable variables
    R = persis_info['R']
    g_k = 1.0 # = 2.0/(k+1) with k=1
    f_i_idxs = persis_info['f_i_idxs']
    num_outer_iters = persis_info['N']

    print('[{}/{}] x={}'.format(0, num_outer_iters, x0), flush=True)

    f_obj = 0

    for k in range(num_outer_iters):

        pre_x_k = (1.0 - g_k) * post_x_k + (g_k * x_k)

        H_o = np.zeros(1, dtype=gen_specs['out'])
        H_o['x'][0] = pre_x_k
        H_o['pt_id'][0] = ct      
        H_o['consensus_pt'][0] = True
        ct += 1

        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        # TODO: Error checking

        # mutable (i.e., changes between outer iters) variables 
        b_k = Work['persis_info']['beta_k']
        g_k = Work['persis_info']['gamma_k']
        T_k = num_inner_iters = Work['persis_info']['T_k']

        assert 'x' in calc_in.dtype.names, print("Alloc did not send neighboring {x_k underscore} as anticipated ...")
        neighbors_pre_x_k = calc_in['x']
        num_neighbors = len(neighbors_pre_x_k)
        # computes local grad_x { (W \otimes I)[x_1,x_2,...,x_m] }
        gradg = 2*R * ( num_neighbors * pre_x_k - np.sum(neighbors_pre_x_k, axis=0) )

        # print('num of inner iters: {}. norm of grad: {:.4e}'.format(T_k, la.norm(gradg)), flush=True)
        x_k, x2_k, ct = PS(x_k, gradg, b_k, T_k, ct, f_i_idxs, gen_specs, libE_info, pre_x_k)

        # recieved end signal 
        if x_k is None:
            return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

        post_x_k = (1.0-g_k) * post_x_k + (g_k * x2_k)

        # guage progress
        l = len(f_i_idxs)
        H_o = np.zeros(l, dtype=gen_specs['out'])
        H_o['x'][:] = post_x_k
        H_o['pt_id'][:] = ct      
        H_o['consensus_pt'][:] = False
        H_o['obj_component'][:] = f_i_idxs
        H_o['get_grad'][:] = False
        H_o = np.reshape(H_o, newshape=(-1,))      
        ct += 1
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, None, ct
        f_is = calc_in['f_i']
        f_obj = np.sum(f_is)
        # gradf_is = calc_in['gradf_i']
        # gradf    = np.sum(gradf_is, axis=0)

        # print('[{}/{}] x={} ||gradf||={:.8f}'.format(k+1, num_outer_iters, post_x_k, la.norm(gradf, ord=2)), flush=True)

        # print('[{}/{}] x={}'.format(k+1, num_outer_iters, post_x_k), flush=True)
        print('[{}: {}/{}] f_is={:.8f}'.format(persis_info['worker_num'], k+1, num_outer_iters, f_obj), flush=True)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def PS(x, gradg, beta, T, ct, f_i_idxs, gen_specs, libE_info, pre_x_k):
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
        H_o['pt_id'][:] = ct      
        H_o['consensus_pt'][:] = False
        H_o['obj_component'][:] = f_i_idxs
        H_o['get_grad'][:] = True
        H_o = np.reshape(H_o, newshape=(-1,))      # unfold into 1d array

        ct += 1
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        if tag in [STOP_TAG, PERSIS_STOP]:
            return None, None, ct

        assert 'gradf_i' in calc_in.dtype.names, print("Alloc did not send \
                `gradf_i` as anticipated ...")

        # TODO: We assume data is in order we requested: verify the code or manually check this 

        p_t = t/2.0
        theta_t = 2.0*(t+1)/(t * (t+3))

        # add gradients since we have sum of convex functions
        gradf_is = calc_in['gradf_i']
        gradf    = np.sum(gradf_is, axis=0)

        u_next = get_l2_argmin(x, u, gradf, gradg, beta, p_t)
        # if np.any(np.isnan(u_next)) or np.any(np.abs(u_next) > 10):
        #     import ipdb; ipdb.set_trace()

        cons = 0.5 * np.dot(pre_x_k, gradg)
        score_u_prev = cons + np.dot(gradg, u-pre_x_k) + np.dot(gradf,u) + beta/2.0*la.norm(x-u, 2)**2
        score_x = cons + np.dot(gradg, x-pre_x_k) + np.dot(gradf,x) + beta*p_t/2.0*la.norm(x-u, 2)**2
        score_u_curr = cons + np.dot(gradg, u_next-pre_x_k) + np.dot(gradf,u_next) + beta/2.0*la.norm(u_next-x,2)**2 + beta*p_t/2.0*la.norm(u_next-u,2)**2
        # print('[Best? {}] u_prev={:.4e}, x={:.4e}, argmin={:.4e}'.format(
        #     score_u_curr <= score_u_prev and score_u_curr <= score_x,
        #     score_u_prev, score_x, score_u_curr), flush=True)

        u = u_next
        u2 = (1-theta_t) * u2 + (theta_t * u)

    return u, u2, ct

def get_l2_argmin(x, u_prev, gradf, gradg, beta, p_t):
    u_next = (beta * x) + (beta * p_t * u_prev) - gradf - gradg
    u_next = u_next / (beta * (1.0+p_t))
    return u_next

def get_entropy_argmin():
    const = 1.0/(1+p_t) * (np.log(x) + (1+p_t) * np.ones(n, dtype=float)) \
            - 2/(beta * (1+p_t))*gradg

    assert len(grad_f) == len(u), print("len(grad_f)={}, expected {}".format(len(grad_f), len(u)))

    dyn = p_t/(1+p_t) * np.log(u) - 1.0/( beta * (1+p_t) ) * grad_f

    u_next = np.exp( const + dyn )

    return u_next
