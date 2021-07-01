import numpy as np

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

def primaldual_slide(H, persis_info, gen_specs, libE_info):
    """ Gradient sliding. Coordinates with alloc to do local and distributed 
        (i.e., gradient of consensus step) calculations.
    """
    # TODO: Allow early termination by checking tag
    tag = None
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(ub)

    # start with random x0
    x_0 = persis_info['rand_stream'].uniform(low=lb, high=ub, size=(n,)) 

    # immutable variables
    mu              = persis_info['mu']
    tau             = persis_info['tau']
    R               = persis_info['R']
    L               = persis_info['L']
    A_norm          = persis_info['A_norm']
    num_outer_iters = persis_info['N']
    f_i_idxs        = persis_info['f_i_idxs']

    lam = tau/(1+tau)
    Delta = np.inf if mu == 0 else int(2*tau+1 + 1) 

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

    for k in range(1,num_outer_iters+1):
        # define parameters
        tau_k = (k-1)/2
        lam_k = (k-1)/k
        b_k   = k
        p_k   = 2*L/k
        T_k   = int(k*R*A_norm/L + 1)

        x_tk = prev_x_k + lam_k*(prev_x_hk - prevprev_x_k)
        x_uk = (x_tk + tau_k*prev_x_uk)/(1+tau_k)

        gradf = get_grad(x_uk, f_i_idxs, gen_specs, libE_info)

        settings = {'T_k': T_k, 
                    'b_k': k, 
                    'p_k': 2*L/k, 
                    'mu': mu, 
                    'L': L,
                    'R': R,
                    'k': k,
                    'prev_b_k': prev_b_k,
                    'prev_T_k': prev_T_k}

        [x_k, x_k_1, z_k, x_hk] = primaldual_slide_inner(gradf, 
                                                  prev_x_k, 
                                                  prev_penult_k,
                                                  prev_z_k, 
                                                  settings,
                                                  gen_specs,
                                                  libE_info)

        prev_prev_x_k = prev_x_k
        prev_x_k      = x_k
        prev_x_hk     = x_hk
        prev_penult_k = x_k_1 # penultimate x_k^{(i)}
        prev_b_k      = b_k
        prev_T_k      = T_k

        weighted_x_hk_sum += b_k * x_hk
        b_k_sum += b_k

    # final solution (weighted average)
    x_star = 1.0/b_k_sum * weighted_x_hk_sum

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def primaldual_slide_inner(y_k, x_curr, x_prev, z_t, settings, gen_specs, libE_info):
    # define params
    T_k = settings['T_k']
    b_k = settings['b_k']
    p_k = settings['p_k']
    mu  = settings['mu']
    L   = settings['L']
    R   = settings['R']
    k   = settings['k']
    prev_b_k = settings['prev_b_k']
    prev_T_k = settings['prev_T_k']

    x_k_1 = x_curr.copy()
    xsum = np.zeros(len(x_curr), dtype=float)
    zsum = np.zeros(len(x_curr), dtype=float)

    for t in range(1,T_k+1):
        # define per-iter params
        eta_t = (p_k + mu)*(t-1) + p_k*T_k
        q_t   = L*T_k/(2*b_k*R**2)
        if k >= 2 and t == 1:
            a_t = prev_b_k*T_k/(b_k*prev_T_k)
        else:
            a_t = 1

        u_t = x_curr + a_t * (x_curr - x_prev)

        # communication #1
        neighbor_u_ts = get_neighbor_vals(u_t, gen_specs, libE_info)
        num_neighbors = len(neighbor_u_ts)
        Lu = (num_neighbors * u_t) - np.sum(neighbor_u_ts, axis=0)

        # compute first argmin
        z_t = z_t - (1.0/q_t) * Lu

        # communication #2
        neighbor_z_ts = get_neighbor_vals(z_t, gen_specs, libE_info)
        Lz = (num_neighbors * z_t) - np.sum(neighbor_z_ts, axis=0)

        # computes second argmin
        x_next = (eta_t*x_curr) + (p_k*x_k_1) - (y_k + Lz)
        x_next /= (eta_t + p_k)

        x_prev = x_curr
        x_curr = x_next

        xsum += x_curr
        zsum += z_t

    x_k   = x_curr
    x_k_1 = x_prev
    z_k   = z_t
    x_hk  = xsum/T_k

    return [x_k, x_k_1, z_k, x_hk]

def get_grad(x, f_i_idxs, gen_specs, libE_info):
    l = len(f_i_idxs)
    H_o = np.zeros(l, dtype=gen_specs['out'])
    H_o['x'][:] = x
    H_o['consensus_pt'][:] = False
    H_o['obj_component'][:] = f_i_idxs
    H_o['get_grad'][:] = True
    H_o = np.reshape(H_o, newshape=(-1,))      # unfold into 1d array

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

    gradf_is = calc_in['gradf_i']
    gradf    = np.sum(gradf_is, axis=0)

    return gradf

def get_neighbor_vals(x, gen_specs, libE_info):
    """ Sends local gen data (@x) and retrieves neighbors local data.
        Sorts the data so the gen ids are in increasing order

    Parameters
    ----------
    x : np.ndarray
        - local input variable

    gen_specs, libE_info : ?
        - objects to communicate and construct mini History array

    Returns
    -------
    X : np.ndarray 
        - 2D array of neighbors and local x values sorted by gen_ids
    """
    H_o = np.zeros(1, dtype=gen_specs['out'])
    H_o['x'][0] = x
    H_o['consensus_pt'][0] = True

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

    neighbor_X = calc_in['x']

    return neighbor_X
