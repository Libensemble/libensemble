import numpy as np
import numpy.linalg as la
import scipy.sparse as spp

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

def grad_track(H, persis_info, gen_specs, libE_info):
    """ Gradient sliding. Coordinates with alloc to do local and distributed 
        (i.e., gradient of consensus step) calculations.
    """
    # TODO: Allow early termination by checking tag
    tag = None
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    n = len(ub)

    # start with random x0
    x0 = persis_info['rand_stream'].uniform(low=lb, high=ub, size=(n,)) 
    x_k = x0

    # immutable variables
    eta             = persis_info['eta']
    num_outer_iters = persis_info['N']
    f_i_idxs        = persis_info['f_i_idxs']
    S_i_data        = persis_info['S_i_data']
    S_i_gen_ids     = persis_info['S_i_gen_ids']
    local_gen_id    = persis_info['worker_num']

    # sort S_i to be increasing gen_id
    _perm_ids = np.argsort(S_i_gen_ids)
    S_weights = S_i_data[_perm_ids]
    S_i_gen_ids = S_i_gen_ids[_perm_ids]
    S_i_gen_ids_no_local = np.delete(S_i_gen_ids, \
                                     np.where(S_i_gen_ids==local_gen_id)[0][0])

    prev_s_is = np.zeros((len(S_i_data),n), dtype=float)
    prev_gradf_is = np.zeros((len(S_i_data),n), dtype=float)

    ones_arr = np.ones(len(x_k), dtype=float)

    for k in range(num_outer_iters):

        gradf = get_grad(x_k, f_i_idxs, gen_specs, libE_info)

        err = la.norm(x_k - ones_arr, ord=2) 
        print('[{}/{}] x={} |gradf|={:4e} abserr={:4e}'.format(k, num_outer_iters, x_k, la.norm(gradf,ord=2), err), flush=True)

        neighbor_gradf_is = get_neighbor_vals(gradf, local_gen_id, \
                                    S_i_gen_ids_no_local, gen_specs, libE_info)

        U = (prev_s_is + neighbor_gradf_is - prev_gradf_is)
        # takes linear combination as described by equation (9)
        s = np.dot(U.T, S_weights)

        neighbor_x_is = get_neighbor_vals(x_k, local_gen_id, S_i_gen_ids_no_local,\
                                          gen_specs, libE_info)
        neighbor_s_is = get_neighbor_vals(s, local_gen_id, S_i_gen_ids_no_local,\
                                          gen_specs, libE_info)

        V = (neighbor_x_is - eta * neighbor_s_is)
        # takes linear combination as described by equation (10)
        next_x_k = np.dot(V.T, S_weights)

        x_k = next_x_k

        # Save data to avoid more communication
        prev_s_is = neighbor_s_is
        prev_gradf_is = neighbor_gradf_is

    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG

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

def get_neighbor_vals(x, local_gen_id, S_gen_ids_no_local, gen_specs, libE_info):
    """ Sends local gen data (@x) and retrieves neighbors local data.
        Sorts the data so the gen ids are in increasing order

    Parameters
    ----------
    x : np.ndarray
        - local input variable

    local_gen_id : int
        - this gen's gen_id

    S_gen_ids_local : int
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
    assert np.array_equal(S_gen_ids_no_local, neighbor_gen_ids),'Expected ' + \
                'gen_ids {}, received {}'.format(S_gen_ids, gen_ids)

    X = np.vstack((neighbor_X, x))
    gen_ids = np.append(neighbor_gen_ids, local_gen_id)

    # sort data (including local) in corresponding gen_id increasing order
    X[:] = X[np.argsort(gen_ids)]

    return X
