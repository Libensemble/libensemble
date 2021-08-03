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
    local_gen_id = persis_info['worker_num']

    # each gen has unique interal id
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(ub)

    f_i_idxs    = persis_info['f_i_idxs']
    A_i_data    = persis_info['A_i_data']
    A_i_gen_ids = persis_info['A_i_gen_ids']
    # sort A_i to be increasing gen_id
    _perm_ids = np.argsort(A_i_gen_ids)
    A_i_weights = A_i_data[_perm_ids]
    A_i_gen_ids = A_i_gen_ids[_perm_ids]
    A_i_gen_ids_no_local = np.delete(A_i_gen_ids, \
                                     np.where(A_i_gen_ids==local_gen_id)[0][0])

    # start with random x_0
    x_0 = persis_info['rand_stream'].uniform(low=lb, high=ub, size=(n,)) 
    # ===== NOTATION =====
    # x_hk == \hat{x}_k
    # x_tk == \tilde{x}_k 
    # x_uk == \underline{x}_k
    # prev_x_k == x_{k-1}
    # prevprev_x_k = x_{k-2}
    # ====================
    prev_x_k     = x_0.copy()
    prev_x_uk    = x_0.copy()
    prev_x_hk    = x_0.copy()
    prev_z_k     = np.zeros(len(x_0), dtype=float)
    prevprev_x_k = x_0.copy()
    prev_penult_k= x_0.copy()

    print_progress = True
    if print_progress:
        print('[{}]: x={}'.format(local_gen_id, x_0), flush=True)

    # TODO: define variables
    mu     = persis_info['params']['mu']
    L      = persis_info['params']['L']
    A_norm = persis_info['params']['A_norm']
    Vx_0x  = persis_info['params']['Vx_0x']
    eps    = persis_info['params']['eps']

    R = 1.0/(4 * (Vx_0x)**0.5)
    N = int(4 * (L*Vx_0x/eps)**0.5 + 1)

    weighted_x_hk_sum = np.zeros(len(x_0), dtype=float)
    b_k_sum = 0

    prev_b_k = 0
    prev_T_k = 0

    for k in range(1,N+1):
        tau_k = (k-1)/2
        lam_k = (k-1)/k
        b_k   = k
        p_k   = 2*L/k
        T_k   = int(k*R*A_norm/L + 1)

        x_tk = prev_x_k + lam_k*(prev_x_hk - prevprev_x_k)
        x_uk = (x_tk + tau_k*prev_x_uk)/(1+tau_k)

        y_k = get_grad(x_uk, f_i_idxs, gen_specs, libE_info)

        settings = {'T_k': T_k, 
                    'b_k': k, 
                    'p_k': 2*L/k, 
                    'mu': mu, 
                    'L': L,
                    'R': R,
                    'k': k,
                    'prev_b_k': prev_b_k,
                    'prev_T_k': prev_T_k,
                    'local_gen_id': local_gen_id,
                    'A_weights': A_i_weights,
                    'A_gen_ids_no_local': A_i_gen_ids_no_local,
                    }

        [x_k, x_k_1, z_k, x_hk] = primaldual_slide(y_k, 
                                                   prev_x_k, 
                                                   prev_penult_k,
                                                   prev_z_k, 
                                                   settings,
                                                   gen_specs, libE_info)

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

        if print_progress:
            curr_x_star = 1.0/b_k_sum * weighted_x_hk_sum
            print('[{}]: x={}'.format(local_gen_id, curr_x_star), flush=True)

    x_star = 1.0/b_k_sum * weighted_x_hk_sum

    if print_progress:
        print('[{}]: x={}'.format(local_gen_id, x_star))
        print_final_score(x_star, f_i_idxs, gen_specs, libE_info)

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

def primaldual_slide(y_k, x_curr, x_prev, z_t, settings, gen_specs, libE_info):

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
    local_gen_id       = settings['local_gen_id'] 
    A_weights          = settings['A_weights']
    A_gen_ids_no_local = settings['A_gen_ids_no_local']

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
        U_ts = get_neighbor_vals(u_t, local_gen_id, A_gen_ids_no_local, 
                                 gen_specs, libE_info)
        Lu_t = np.dot(U_ts.T, A_weights)
        z_t = z_t + (1.0/q_t) * Lu_t

        # computes second argmin
        Z_ts = get_neighbor_vals(z_t, local_gen_id, A_gen_ids_no_local, 
                                 gen_specs, libE_info)
        Lz_t = np.dot(Z_ts.T, A_weights)
        x_next = (eta_t*x_curr) + (p_k*x_k_1) - (y_k + Lz_t)
        x_next /= (eta_t + p_k)

        x_prev = x_curr
        x_curr = x_next

        xsum += x_curr
        # zsum += z_t

    x_k   = x_curr
    x_k_1 = x_prev
    z_k   = z_t
    x_hk  = xsum/T_k

    return [x_k, x_k_1, z_k, x_hk]

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

   sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

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

def get_neighbor_vals(x, local_gen_id, A_gen_ids_no_local, gen_specs, libE_info):
    """ Sends local gen data (@x) and retrieves neighbors local data.
        Sorts the data so the gen ids are in increasing order

    Parameters
    ----------
    x : np.ndarray
        - local input variable

    local_gen_id : int
        - this gen's gen_id

    A_gen_ids_local : int
        - expected neighbor's gen ids, not including local gen id

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
    neighbor_gen_ids = calc_in['gen_worker']

    assert local_gen_id not in neighbor_gen_ids, 'Local data should not be ' + \
                                                 'sent back from manager'
    assert np.array_equal(A_gen_ids_no_local, neighbor_gen_ids),'Expected ' + \
                'gen_ids {}, received {}'.format(A_gen_ids, gen_ids)

    X = np.vstack((neighbor_X, x))
    gen_ids = np.append(neighbor_gen_ids, local_gen_id)

    # sort data (including local) in corresponding gen_id increasing order
    X[:] = X[np.argsort(gen_ids)]

    return X
