import numpy as np
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
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
        alg_vars = define_alg_vars(alloc_specs, gen_specs, persis_info)
        persis_info['first_call'] = False
        persis_info['alg_vars'] = alg_vars  # parameters needed for optimization alg
        persis_info['outer_iter_ct'] = 1          # number of outer iterations

    # Exit if all persistent gens are done
    elif gen_count == 0:
        return Work, persis_info, 1

    # Exit once we have exceeded outer iteration count
    if persis_info['outer_iter_ct'] > persis_info['alg_vars']['N']+1:
        # TODO: Average work? Or take the minimum of them all...
        # TODO: Send signal to shut gens down?
        return Work, persis_info, 1

    m = alloc_specs['user']['m']
    num_gens_done_with_ps = 0
    # Sort to give consistent ordering for gens
    avail_persis_worker_ids = np.sort( avail_worker_ids(W, persistent=True) )

    # Give completed gradients back to gens
    for gen_id in avail_persis_worker_ids:

        # if no prox steps, gen is waiting for consensus gradient 
        if persis_info[gen_id]['T_k'] == 0:
            num_gens_done_with_ps += 1
            continue

        ret_sim_idxs_from_gen_i = np.where( 
                np.logical_and.reduce(( 
                    H['returned'], 
                    ~H['ret_to_gen'], 
                    ~H['consensus_pt'], 
                    H['gen_worker']==gen_id 
                )))[0]

        if len(ret_sim_idxs_from_gen_i) == 0: 
            continue

        pt_ids_from_gen_i = set(H[ret_sim_idxs_from_gen_i]['pt_id'])
        sim_ids_to_send = np.array([], dtype=int) 

        for pt_id in pt_ids_from_gen_i:

            ret_sim_idxs_with_pt_id = ret_sim_idxs_from_gen_i[ 
                    H[ret_sim_idxs_from_gen_i]['pt_id'] == pt_id ]

            # TODO: This is a brute force fix, can we dynamically get the size
            num_req_grad_computations = len(gen_specs['user']['lb']) * \
                    len(persis_info[gen_id].get('f_i_idxs', []))

            assert len(ret_sim_idxs_with_pt_id) <= num_req_grad_computations, \
                    "Recieved {} sim data pts, expected at most {}".format( 
                        len(returned_pt_id_sim_idxs), m)

            if len(ret_sim_idxs_with_pt_id) == num_req_grad_computations:
                # if we have collected all gradient queries, marks a single prox step
                persis_info[gen_id]['T_k'] -= 1

                sim_ids_to_send = np.append(sim_ids_to_send, ret_sim_idxs_with_pt_id )
                # IMPORTANT: first index by ['ret_to_gen'] to avoid copy
                H['ret_to_gen'][ ret_sim_idxs_with_pt_id ] = True 

        if len(sim_ids_to_send) > 0:
            gen_work(Work, 
                     gen_id, 
                     ['gradf_i_x_j'], 
                     np.atleast_1d(sim_ids_to_send), 
                     persis_info.get(gen_id), 
                     persistent=True)

    # If all x's collected, help gens compute $(W \otimes I) [x_1, x_2, ...]$
    if num_gens_done_with_ps == alloc_specs['user']['num_gens']:

        consensus_idx_arr = np.zeros(num_gens_done_with_ps, dtype=int)

        for i, gen_id in enumerate(avail_persis_worker_ids):

            incomplete_consensus_idxs = np.where(
                np.logical_and.reduce((~H['ret_to_gen'], H['consensus_pt'], H['gen_worker'] == gen_id)))[0]
            assert len(incomplete_consensus_idxs) == 1, print(
                'gen_id={} recieved {} incomplete consensus points, expected 1'.format(
                        gen_id, len(incomplete_consensus_idxs) 
                ))

            consensus_idx_arr[i] = incomplete_consensus_idxs[-1] # last incomplete req
            # H[last_idx_from_gen_i]['given'] = True

        k = persis_info['outer_iter_ct'] 
        A = persis_info['alg_vars']['A']
        L = persis_info['alg_vars']['L']
        N = persis_info['alg_vars']['N']
        M = persis_info['alg_vars']['M']
        D = persis_info['alg_vars']['D']
        R = persis_info['alg_vars']['R']
        nu = persis_info['alg_vars']['nu']

        b_k = 2.0*L/(nu * k)
        g_k = 2.0/(k+1)
        T_k = int( (N*((M*k)**2)) / (D*(L**2)) + 1 )

        # send neighbors' {x_k underscore} between gens and prepare for new outer iter
        for i, gen_id in enumerate(avail_persis_worker_ids):
            
            persis_info[gen_id]['T_k'] = T_k
            persis_info[gen_id]['beta_k'] = b_k
            persis_info[gen_id]['gamma_k'] = g_k

            incident_gens = A.indices[ A.indptr[i]:A.indptr[i+1] ]
            assert i not in incident_gens, print("no self loops permiited in \
                    adjacency matrix @A (i.e. only zeros on diagonal)")
            neighbor_consensus_idx = consensus_idx_arr[ incident_gens ]

            gen_work(Work, gen_id, ['x'], np.atleast_1d(neighbor_consensus_idx),
                     persis_info.get(gen_id), persistent=True)

            curr_gen_consensus_idx = consensus_idx_arr[i]
            H[curr_gen_consensus_idx]['ret_to_gen'] = True

        persis_info['outer_iter_ct'] += 1

    # partition sum of convex functions evenly (only do at beginning)
    if persis_info['outer_iter_ct'] == 1 and len( avail_worker_ids(W, persistent=False) ):
        num_funcs_arr = partition_funcs_arr(alloc_specs['user']['m'], 
                alloc_specs['user']['num_gens'])
        k = persis_info['outer_iter_ct'] 
        b_k = 2.0*persis_info['alg_vars']['L']/k
        g_k = 2.0/(k+1)

    # TODO: What if task is put in Work but not marked given yet?
    task_avail = np.logical_and(~H['given'], ~H['consensus_pt'])

    for i in avail_worker_ids(W, persistent=False):

        # start up gens
        if persis_info['outer_iter_ct'] == 1 and gen_count < alloc_specs['user']['num_gens']:
            gen_count += 1
            l_idx = num_funcs_arr[gen_count-1]
            r_idx = num_funcs_arr[gen_count]

            # We need gradient of consensus before proximal-slide, so tell gen
            # we have "finished" proximal slide by setting num_steps_left=0

            persis_info[i].update({
                'f_i_idxs': range(l_idx, r_idx),
                'T_k': 0, 
                'beta_k': b_k,
                'gamma_k': g_k,
                'R':  persis_info['alg_vars']['R'],
                'N':persis_info['alg_vars']['N'],
                })
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

def define_alg_vars(alloc_specs, gen_specs, persis_info):
    """ Variables for prox-slide algorithm 
    """
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    b = gen_specs['user']['gen_batch_size']
    m = alloc_specs['user']['m']  
    f_i_idxs = persis_info.get('f_i_idxs')

    n = len(lb)

    # User defined
    M   = 10    # upper bound on gradient
    eps = 10    # error/tolerance
    R   = 100   # consensus penalty
    D_X = 50*n   # diameter of X
    nu  = 2     # modulus of strong convexity of \omega

    D = (3*D_X)/(2*nu)

    # chain matrix
    num_gens = alloc_specs['user']['num_gens']
    diagonals = [np.ones(num_gens-1), np.ones(num_gens-1)]
    A = spp.csr_matrix( spp.diags(diagonals, [-1,1]) )

    if num_gens > 2:
        lam_max = sppla.eigs(A, k=1)[0][0]
    else:
        import numpy.linalg as la
        lam_max = np.amax( la.eigvals(A.todense()) )
    L = 2*R*lam_max
    N = int( ((L * D_X)/(nu * eps) )**0.5 + 1 )

    alg_vars = {
                'M': M,               # upper bound on gradient of f
                'L': L,               # L-smoothness of consensus
                'D': D,               # (3 D_X^2)/(2 nu), where D_X is the diameter of X 
                'A': A,               # Adjacency matrix (we will not explicitly form Laplacian)
                'R': R,               # consensus penalty, represents R=R_y^2/eps
                'N': N,               # number of outer iterations 
                'nu': nu,             # modulus of strong convexity
                }

    return alg_vars

def partition_funcs_arr(num_funcs, num_gens):
    num_funcs_arr = (num_funcs//num_gens) * np.ones(num_gens, dtype=int)
    num_leftover_funcs = num_funcs % num_gens
    num_funcs_arr[:num_leftover_funcs] += 1

    # builds starting and ending function indices for each gen e.g. if 7
    # functions split up amongst 3 gens, then num_funcs__arr = [0, 3, 5, 7]
    num_funcs_arr = np.append(0, np.cumsum(num_funcs_arr))

    return num_funcs_arr
