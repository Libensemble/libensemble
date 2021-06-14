import numpy as np
import scipy.sparse as spp
from libensemble.tools.alloc_support import (avail_worker_ids, sim_work, gen_work,
                                             count_persis_gens, all_returned)


def start_proxslide_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation will start gens to solve a distributed optimization problem
    (see https://arxiv.org/pdf/1911.10645) where x is a @n dimensional vector in

    \begin{equation}
        \min_x \sum\limits_{i=1}^m f_i(x).
    \end{equation}

    To do so, the method, roughly speaking, solves @m separate problem

    \begin{equation}
        \min_{x_1, ..., x_m} \sum\limits_{i=1}^m f_i(x_i)
    \end{equation}

    with a penalty $||W[x_1,...,x_m]||$ that is 0 when $x_1=x_2=...=x_m$,
    is low if the variables are similar, but is larger otherwise, where W
    corresponds to a graph that maps the network between gen workers via a
    graph (graph can be any connected graph, such as a star, chain, cycle,
    complete graph, etc.). We call ||W[x_1,...]|| the consensus score since it
    provides a scalar score of how similar the independent subproblems (i.e.,
    x_i) are to one another.  
    Overall, the distribuetd algorithm has 4 stages:

    1. pre-add (computes x_k underscore)
    2. gradient consesus score (requires communication between certain nodes)
    3. proximal-slide (requires gradient computation)
    4. post add (computes x_k overscore)

    The role of alloc is to 1) start gens, 2) receive and distributed {x_k underscore} 
    to necessary gens (i.e., compute gradient of consensus via one communication
    round) and 3) queue sims for gradient computaitons requested by gen and
    return once done. Note that the three steps that are not consensus-related
    can be done locally in each gen (see file
    `libensemble.gen_funcs.persistent_prox_slide`).
    """

    Work = {}
    gen_count = count_persis_gens(W)

    if persis_info.get('first_call', True):
        alg_vars = define_alg_vars(gen_specs, persis_info)
        persis_info['first_call'] = False
        persis_info['alg_vars'] = alg_vars  # parameters needed for optimization alg
        persis_info['iter_ct'] = 1          # number of outer iterations

    # Exit if all persistent gens are done
    elif gen_count == 0:
        return Work, persis_info, 1

    # Exit once we have exceeded outer iteration count
    if persis_info['iter_ct'] > persis_info['alg_vars']['N']:
        # TODO: Average work? Or take the minimum of them all...
        # TODO: Send signal to shut gens down?
        return Work, persis_info, 1

    m = gen_specs['user']['m']
    num_gens_done_with_ps = 0
    avail_persis_worker_ids = np.sort( avail_worker_ids(W, persistent=True) )

    for i in avail_persis_worker_ids:

        # if no more prox steps, gen is waiting for gradient of consensus
        # (requires all gens to be fininshed) and the gen does need sims yet
        if persis_info[i]['num_prox_steps_left'] > 0:
            num_gens_done_with_ps += 1
            continue
        else:
            persis_info[i]['num_prox_steps_left'] -= 1

        ret_sim_idxs_from_gen_i = np.where( 
                np.logical_and(H['returned'], 
                    np.logical_and(~H['ret_to_gen'], H['gen_worker']==i ))
                )[0]

        pt_ids_from_gen_i = set(H[ret_sim_idxs_from_gen_i]['pt_id'])

        if len(pt_ids_from_gen_i) == 0: continue

        root_idxs = np.array([], dtype=int) # which history idxs have sum_i f_i

        for pt_id in pt_ids_from_gen_i:

            # TODO: Can we do consecutive accesses, e.g. tuple (10,44) vs arr [10,11,...,44]?
            subset_sim_idxs = np.where( H[ret_sim_idxs_from_gen_i]['pt_id'] == pt_id )[0]
            ret_sim_idxs_with_pt_id = ret_sim_idxs_from_gen_i[ subset_sim_idxs ]

            assert len(ret_sim_idxs_with_pt_id) <= m, \
                    "{} incorrect number of sim data pts, expected {}".format( 
                        len(returned_pt_id_sim_idxs), m)

            if len(ret_sim_idxs_with_pt_id) == m:

                returned_fvals = H[ ret_sim_idxs_with_pt_id ]['gradf_i']
                root_idxs = np.append(root_idxs, ret_sim_idxs_with_pt_id )
                # accessing ['ret_to_gen'] is to ensure we do not write to cpy
                H['ret_to_gen'][ ret_sim_idxs_with_pt_id ] = True 

        if len(root_idxs) > 0:
            import ipdb; ipdb.set_trace()
            gen_work(Work, i, ['gradf_i'], np.atleast_1d(root_idxs), persis_info.get(i), persistent=True)

    all_gens_done_with_ps = num_gens_done_with_ps == alloc_specs['user']['num_gens']
    # Implicitly computes $(W \otimes I) [x_1, x_2, ..., x_m]$
    if all_gens_done_with_ps:

        """
        n = len(gen_specs['user']['lb'])
        X = np.empty(shape=(num_gens_done_with_ps, n), dtype=float)

        for i, gen_id in enumerate(avail_persis_worker_ids):

            # last request vector from gen is assumed to be {x_k underscore}
            idxs_todo_from_gen = H[ ~H['given'] ][ H['gen_id'] == gen_id ]
            assert len(idxs_todo_from_gen), print("gen did not send {x_k underscore}")

            last_idx_from_gen = idxs_todo_from_gen[-1]
            x_k = H[last_idx_from_gen]['x']
            X[i] = x_k
            H[last_idx_from_gen]['given'] = True
            H[last_idxs_from_gen]['returned'] = True

        R_y = persis_info['alg_vars']['R_y']
        eps = persis_info['alg_vars']['eps']
        W_consensus = persis_info['alg_vars']['W']
        x = np.reshape(X, newshape=(-1,))                               # unfold 
        consensus_grad = 2*R_y/eps * W_consesus.dot(x)
        consensus_grad = np.reshape( consensus_grad, newshape=X.shape ) # refold
        """

        # construct array that has each gen's last index in History array, 
        # which corresponds to where {x_k underscore} is stored
        gen_last_H_idx = np.zeros(num_gens_done_with_ps)

        for i, gen_id in enumerate(avail_persis_worker_ids):
            idxs_todo_from_gen = H[ ~H['given'] ][ H['gen_id'] == gen_id ]
            assert len(idxs_todo_from_gen), print("gen did not send {x_k underscore}")
            last_idx_from_gen_i = idxs_todo_from_gen[-1]
            gen_last_H_idx[i] = last_idx_from_gen_i
            H[last_idx_from_gen_i]['given'] = True

        A = persis_info['alg_vars']['A']
        k = persis_info['alg_vars']['iter_ct'] 
        L = persis_info['alg_vars']['L']
        N = persis_info['alg_vars']['N']
        M2 = persis_info['alg_vars']['M2']
        D2 = persis_info['alg_vars']['D2']
        R_y = persis_info['alg_vars']['R_y']
        eps = persis_info['alg_vars']['eps']
        sigma_sq = persis_info['alg_vars']['sigma_sq']

        b_k = 2.0*L/k
        g_k = 2.0/(k+1)
        T_k = int( N*(M2**2 + sigma_sq)*k**2 / (D2*L**2) )
        T_k = min(10, max(1, T_k//10))

        # send neighbors' {x_k underscore} between gens to implicitly compute
        # $(Wbar \otimes I)[x_1,x_2,...,x_m]$ via adjacency matrix @A
        for i, gen_id in enumerate(avail_persis_worker_ids):
            
            persis_info[gen_id]['num_prox_steps_left'] = T_k
            persis_info[gen_id]['beta_k'] = b_k
            persis_info[gen_id]['gamma_k'] = g_k

            neighrbo_gens = A.indices[ A.indptr[i]:A.indptr[i+1] ]
            assert i not in neighbor_gens_last_H_idx, print("adjacency matrix @A must cannot have nonzero on diagonal")
            neighbor_gens_last_H_idx = neighbor_last_H_idx[ incident_gens ]

            import ipdb; ipdb.set_trace()
            gen_work(Work, gen_id, ['x'], np.atleast_1d(neighbor_gens_last_H_idx), 
                     persis_info.get(i), persistent=True)

            curr_gen_last_H_idx = gen_last_H_idx[i]
            H[curr_gen_last_H_idx]['returned'] = True

        persis_info['iter_ct'] += 1

    num_req_gens = alloc_specs['user']['num_gens']
    # partition sum of convex functions evenly (only do at beginning)
    if persis_info['iter_ct'] == 1 and len( avail_worker_ids(W, persistent=False) ):
        num_funcs_arr = partition_funcs_evenly_as_arr(alloc_specs['user']['num_gens'], num_req_gens)
        k = persis_info['iter_ct'] 
        b_k = 2.0*persis_info['alg_vars']['L']/k
        g_k = 2.0/(k+1)

    # TODO: What if task is put in Work but not market given yet?
    task_avail = ~H['given'] 

    for i in avail_worker_ids(W, persistent=False):

        # start up gens
        if persis_info['iter_ct'] == 1 and gen_count < alloc_specs['user']['num_gens']:
            gen_count += 1
            l_idx = num_funcs_arr[gen_count-1]
            r_idx = num_funcs_arr[gen_count]

            # We need gradient of consensus before proximal-slide, so tell gen
            # we have "finished" proximal slide by setting num_steps_left=0

            persis_info[i].update({
                'f_i_idxs': range(l_idx, r_idx),
                'num_prox_steps_left': 0, 
                'beta_k': b_k,
                'gamma_k': g_k,
                'R_y':  persis_info['alg_vars']['R_y'],
                'eps': persis_info['alg_vars']['eps'], 
                'N':persis_info['alg_vars']['N'],
                })
            import ipdb; ipdb.set_trace()
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True)

        # give sim work when task available 
        elif np.any(task_avail):
            q_inds = 0 # start with oldest point in queue

            sim_ids_to_send = np.nonzero(task_avail)[0][q_inds]  
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info.get(i))

            task_avail[sim_ids_to_send] = False

        else:
            break

    return Work, persis_info, 0

def define_alg_vars(gen_specs, persis_info):
    """ Variables for prox-slide algorithm 
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    m = gen_specs['user']['m']  
    n = len(lb)
    b = gen_specs['user']['gen_batch_size']
    f_i_idxs = persis_info.get('f_i_idxs')

    C_1 = 1      # see paper
    C_3 = n**0.5 # C_3=1 if 2-norm or sqrt(n) if 1-norm
    Delta = 0    # bound on noise (but we have exact gradient)

    # Potentially user defined
    C = 1        
    M = 100 
    L = 1        
    c = 1        
    eps = 0.1
    # end of user defined

    D_X = 2
    D_XV = (2*np.log(n))**0.5  # diameter of X w.r.t. Bregman divg. V
    D2  = 0.75 * D_XV**2
    s = D_X                    # s <= D_X

    p_star = np.log(n)/n  
    r = 0.5 * s * C_3     
    M2 = c*(n**0.5)*C_1*M
    sigma_sq = 4 * p_star**2 * (C*n*M**2 + (n*Delta/r)**2)

    # chain matrix
    # TODO: Define different types of matrices
    diagonals = [np.ones(m-1), np.ones(m-1)]
    A = spp.csr_matrix( spp.diags(diagonals, [-1,1]) )
    # W = spp.kron(Wbar, spp.eye(n)) 
    lam_min = eps
    R_y = ( M**2/(m*lam_min) )**0.5 # TODO: This might be incorrect ... replace m with number of local functions

    N = 10      # number of iterations to reach desired accuracy, just pick some random number for now

    alg_vars = {
                'C': C,               # = O(1) independent of ...
                'C_1': C_1,           #
                'C_3': C_3,           # 
                'M': M,               # upper bound on ||f'(x)|| 
                'L': L,               # g=2R_y/eps ||Wx|| is L-smooth
                'c' : c,              # c = O(1) independent of n, C_1
                'eps': eps,           # tolerance
                'Delta': Delta,       # ? 
                'p_star': p_star,     # (E[||e||*^4)**0.25 <= p_star ; =1 if 2-norm or O(sqrt( ln(n)/n )) for 1-norm
                'r': r,               # smoothing paramter
                'M2': M2,             # ?
                'D2': D2,             # ?
                'sigma_sq': sigma_sq, # ?
                'A': A,               # Adjacency matrix (we will not explicitly form Laplacian)
                'lam_min': lam_min,   # ?
                'R_y': R_y,           # consensus penalty
                'N': N,               # number of outer iterations 
                }

    return alg_vars

def partition_funcs_evenly_as_arr(num_funcs, num_gens):
    num_funcs_arr = (num_funcs//num_gens) * np.ones(num_gens, dtype=int)
    num_leftover_funcs = num_funcs % num_gens
    num_funcs_arr[:num_leftover_funcs] += 1

    # builds starting and ending function indices for each gen e.g. if 7
    # functions split up amongst 3 gens, then num_funcs__arr = [0, 3, 5, 7]
    num_funcs_arr = np.append(0, np.cumsum(num_funcs_arr))

    return num_funcs_arr
