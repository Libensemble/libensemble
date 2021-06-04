import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

def persistent_smart(H, persis_info, gen_specs, libE_info):
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

    while 1:
        # start with random x0
        x0 = persis_info['rand_stream'].uniform(lb, ub, (1, n))

        post_x = x0
        x = x0

        for k in range(N):

            pre_x = (1-g_k) * post_x + (g_k * x)
        
            x, x2, ct = PS(pre_x, x, b_k, T_k, ct)

            if x is None:
                return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

            post_x = (1-g_k) * post_x + (g_k * x2)


    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def PS(x_center, x, beta, T, ct):
    """ Prox-sliding procedure (see https://arxiv.org/pdf/1911.10645)
    """

    u = x
    u2 = x
    n = len(x)

    Wx_center = np.dot(W, x_center) # one communication round

    for t in range(T):

        # request gradient
        H_o = np.zeros(m, dtype=gen_specs['out'])
        x_req = persis_info['rand_stream'].uniform(lb, ub, (1, n))
        H_o['x'][:] = np.tile(x_req, (m, 1)) # duplicate `x` @m times
                                             # TODO: If `x` is large, can we ref it
        H_o['pt_id'][:] = ct                 # every @m evals is for a single x_i
        H_o['obj_component'][:] = np.arange(0,m)
        ct += 1
        tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

        if tag not in [STOP_TAG, PERSIS_STOP]:
            return None, None, ct

        const = 1.0(1+p_t) * (np.log(x) + (1+p_t) * np.ones(n, dtype=float)) \
                - (2*R_y**2)/(beta * (1-p_t) * eps)*Wx_center

        grad_f = calc_in['gradf']

        dyn = p_t/(1+p_t) * np.log(u) + 1.0/( beta * (1+p_t) ) * grad_f

        u_next = np.exp( const + dyn )

        # this is last line for argmin computation (returns vector in 1-simplex)
        u = u_next/np.sum(u_next) 

        u2 = (1-theta_t) * u2 + (theta_t * u)

    return u, u2, ct
