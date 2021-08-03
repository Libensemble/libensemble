import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

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

def get_consensus_gradient(x, gen_specs, libE_info):
    """ Sends local gen data (@x) and retrieves neighbors local data,
        and takes sum of the neighbors' x's, which is equivalent to taking
        the gradient of consensus term for this particular node/agent. 

        This function is equivalent to the @get_neighbor_vals function, but is
        less general, i.e., when we need only take a sum rather than a linear
        combination of our neighbors.

    Parameters
    ----------
    x : np.ndarray
        - local input variable

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
    num_neighbors = len(neighbor_X)

    return (num_neighbors*x) - np.sum(neighbor_X, axis=0)

def get_k_reach_chain_matrix(n, k):
    """ Constructs adjacency matrix for a chain matrix where the ith vertex can
        reach vertices that are at most @k distances from them (does not wrap around),
        where the distance is based on the absoluate difference between vertices'
        indexes.
    """
    assert 1 <= k <= n-1

    half_of_diagonals = [np.ones(n-k+j) for j in range(k)]
    half_of_indices = np.arange(1,k+1)
    all_of_diagonals = half_of_diagonals + half_of_diagonals[::-1]
    all_of_indices = np.append(-half_of_indices[::-1], half_of_indices)
    A = spp.csr_matrix( spp.diags(all_of_diagonals, all_of_indices) )
    return A

def get_doubly_stochastic(A):
    """ Generates a doubly stochastic matrix where
    (i) S_ii > 0 for all i
    (ii) S_ij > 0 if and only if (i,j) \in E

    Parameter
    ---------
    A : np.ndarray
        - adjacency matrix

    Returns
    -------
    x : scipy.sparse.csr_matrix
    """
    np.random.seed(0)
    n = A.shape[0]
    x = np.multiply( A.toarray() != 0,  np.random.random((n,n)))
    x = x + np.diag(np.random.random(n) + 1e-4)

    rsum = np.zeros(n)
    csum = np.zeros(n)
    tol=1e-15

    while (np.any(np.abs(rsum - 1) > tol)) | (np.any(np.abs(csum - 1) > tol)):
        x = x / x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)

    X = spp.csr_matrix(x)
    return X

