import numpy as np
import numpy.linalg as la
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
        persis_info['first_call'] = False

        alg_vars = define_alg_vars(alloc_specs, gen_specs, persis_info)
        persis_info['alg_vars'] = alg_vars  # parameters needed for optimization alg
        persis_info['outer_iter_ct'] = 1    # number of outer iterations

    # Exit if all persistent gens are done
    elif gen_count == 0:
        return Work, persis_info, 1

    m = alloc_specs['user']['m']
    num_gens_at_consensus = 0
    # Sort to give consistent ordering for gens
    avail_persis_worker_ids = np.sort( avail_worker_ids(W, persistent=True) )

    # Give completed gradients back to gens
    for i in avail_persis_worker_ids:

        # if waiting on consensus, wait until everyone is done
        if persis_info[i].get('at_consensus', False):
            num_gens_at_consensus += 1
            continue

        # gen is waiting on points 
        elif len(persis_info[i].get('curr_H_ids', [])):

            [l_H_id, r_H_id] = persis_info[i].get('curr_H_ids')
            num_sims_req = r_H_id - l_H_id

            num_fin_sims = np.sum(H['returned'][l_H_id:r_H_id])

            # if completed all work, send back
            if num_fin_sims == num_sims_req:

                sims_to_ret_to_gen = np.arange(l_H_id, r_H_id)

                gen_work(Work, i, 
                         ['x', 'f_i', 'gradf_i'], 
                         sims_to_ret_to_gen,
                         persis_info.get(i), 
                         persistent=True)

                persis_info[i].update({'curr_H_ids': []})

        # otherwise, check if gen has requested new work
        else:
            last_H_len = persis_info['last_H_len']

            # first check if gen requested consensus points
            consensus_sim_ids = np.where( 
                np.logical_and( 
                    H[last_H_len:]['consensus_pt'], 
                    H[last_H_len:]['gen_worker']==i )
                )[0] 

            if len(consensus_sim_ids):
            
                assert len(consensus_sim_ids)==1, print('Gen should only send one point for consensus step, received {}'.format(len(consensus_sim_ids)))

                # re-orient
                sim_id = consensus_sim_ids[0] + last_H_len
                persis_info[i].update({'curr_H_ids': [sim_id, sim_id+1]})
                persis_info[i].update({'at_consensus': True})
                num_gens_at_consensus += 1

            # otherwise, sim requested work for sim
            else:

                new_H_ids_from_gen_i = np.where( H[last_H_len:]['gen_worker'] == i )[0]

                assert len(new_H_ids_from_gen_i), print("Gen must request new sim work or show convergence if avail, but neither occured")

                # re-orient (since the last_H_len has relative index 0)
                new_H_ids_from_gen_i += last_H_len

                l_H_id = new_H_ids_from_gen_i[0]
                r_H_id = new_H_ids_from_gen_i[-1] + 1

                assert len(new_H_ids_from_gen_i) == r_H_id - l_H_id, print("new gen data must be in contiguous space")

                persis_info[i].update({'curr_H_ids': [l_H_id, r_H_id] })

    # If all gens at consensus, help compute $(W \otimes I) [x_1, x_2, ...]$
    if num_gens_at_consensus == alloc_specs['user']['num_gens']:

        assert num_gens_at_consensus == len(avail_persis_worker_ids), print(
                'All gens must be available, only {}/{} are though...'.format(
                    len(avail_worker_ids), len(num_gens_at_consensus)))

        k = persis_info['outer_iter_ct'] 
        A = persis_info['alg_vars']['A']
        L = persis_info['alg_vars']['L']
        N = persis_info['alg_vars']['N']
        M = persis_info['alg_vars']['M']
        D = persis_info['alg_vars']['D']
        nu = persis_info['alg_vars']['nu']

        b_k = 2.0*L/(nu * k)
        g_k = 2.0/(k+1)
        T_k = int( (N*((M*k)**2)) / (D*(L**2)) + 1 )

        # compile location of x locations needed for Wx ...
        consensus_ids_in_H = np.array([persis_info[i]['curr_H_ids'][0] for i in avail_persis_worker_ids], dtype=int)
        # TEMP
        num_gens = alloc_specs['user']['num_gens']
        n = len(gen_specs['user']['lb'])
        gradg = np.zeros(num_gens*n, dtype=float)
        x     = np.zeros(num_gens*n, dtype=float)

        # send neighbors' {x_k underscore} between gens and prepare for new outer iter
        for i0, i in enumerate(avail_persis_worker_ids):
            
            persis_info[i]['T_k'] = T_k
            persis_info[i]['beta_k'] = b_k
            persis_info[i]['gamma_k'] = g_k

            incident_gens = A.indices[ A.indptr[i0]:A.indptr[i0+1] ]
            assert i0 not in incident_gens, print("no self loops permiited in \
                    adjacency matrix @A (i.e. only zeros on diagonal)")
            neighbor_consensus_ids_in_H = consensus_ids_in_H[ incident_gens ]

            # TEMP
            x[i0*n:(i0+1)*n] = H[consensus_ids_in_H[i0]]['x']
            gradg[i0*n:(i0+1)*n] = H[consensus_ids_in_H[i0]]['x'] - np.sum(H[neighbor_consensus_ids_in_H]['x'], axis=0)

            gen_work(Work, i, ['x'], np.atleast_1d(neighbor_consensus_ids_in_H),
                     persis_info.get(i), persistent=True)

            persis_info[i].update({'curr_H_ids': []})
            persis_info[i].update({'at_consensus': False})

        # TEMP
        R = persis_info['alg_vars']['R']
        print('|gradg|={:.8f}\n'.format(R * np.dot(x,gradg)), flush=True)
        persis_info['outer_iter_ct'] += 1

    # partition sum of convex functions evenly (only do at beginning)
    if persis_info['outer_iter_ct'] == 1 and len( avail_worker_ids(W, persistent=False) ):
        num_funcs_arr = partition_funcs_arr(alloc_specs['user']['m'], 
                alloc_specs['user']['num_gens'])
        k = persis_info['outer_iter_ct'] 
        b_k = 2.0*persis_info['alg_vars']['L']/k
        g_k = 2.0/(k+1)

    for i in avail_worker_ids(W, persistent=False):

        # start up gens
        if persis_info['outer_iter_ct'] == 1 and gen_count < alloc_specs['user']['num_gens']:
            gen_count += 1
            l_idx = num_funcs_arr[gen_count-1]
            r_idx = num_funcs_arr[gen_count]

            persis_info[i].update({
                'f_i_idxs': range(l_idx, r_idx),
                'T_k': 0, 
                'beta_k': b_k,
                'gamma_k': g_k,
                'R':  persis_info['alg_vars']['R'],
                'N':  persis_info['alg_vars']['N'],
                'at_consensus': False,
                'curr_H_ids': [],
                })
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True)

        # give sim work when task available 
        elif persis_info['next_to_give'] < len(H):

            # skip points that are not sim work or are already done
            while persis_info['next_to_give'] < len(H) and \
                  (H[persis_info['next_to_give']]['given'] or \
                   H[persis_info['next_to_give']]['consensus_pt']):

                persis_info['next_to_give'] += 1

            if persis_info['next_to_give'] >= len(H):
                break

            sim_work(Work, i, 
                     sim_specs['in'], 
                     np.array([persis_info['next_to_give']]),
                     persis_info.get(i))

            persis_info['next_to_give'] += 1

        else:
            break

    persis_info.update({'last_H_len' : len(H)})

    return Work, persis_info, 0

def define_alg_vars(alloc_specs, gen_specs, persis_info):
    """ Variables for prox-slide algorithm 
    """
    # b = gen_specs['user']['gen_batch_size']
    ub = gen_specs['user']['ub']
    lb = gen_specs['user']['lb']
    m = alloc_specs['user']['m']  
    f_i_idxs = persis_info.get('f_i_idxs')

    n = len(lb)

    M   = persis_info['hyperparams']['M']
    R   = persis_info['hyperparams']['R']
    nu  = persis_info['hyperparams']['nu']
    eps = persis_info['hyperparams']['eps']
    D_X = persis_info['hyperparams']['D_X']
    L_const = persis_info['hyperparams']['L_const']
    N_const = persis_info['hyperparams']['N_const']

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
    L = L_const*R*lam_max
    N = N_const * int( ((L * D_X)/(nu * eps) )**0.5 + 1 )

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
