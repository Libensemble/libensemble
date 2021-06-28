import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
from libensemble.tools.alloc_support import (avail_worker_ids, sim_work, gen_work,
                                             count_persis_gens, all_returned)

def start_gradtrack_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
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
            
                assert len(consensus_sim_ids)==1, 'Gen should only send one ' + \
                        'point for consensus step, received {}'.format(len(consensus_sim_ids))

                # re-orient
                sim_id = consensus_sim_ids[0] + last_H_len
                persis_info[i].update({'curr_H_ids': [sim_id, sim_id+1]})
                persis_info[i].update({'at_consensus': True})
                num_gens_at_consensus += 1

            # otherwise, gen requested work for sim
            else:

                new_H_ids_from_gen_i = np.where( H[last_H_len:]['gen_worker'] == i )[0]

                assert len(new_H_ids_from_gen_i), 'Gen must request new sim ' + \
                        'work or show convergence if avail, but neither occured'

                # re-orient (since the last_H_len has relative index 0)
                new_H_ids_from_gen_i += last_H_len

                l_H_id = new_H_ids_from_gen_i[0]
                r_H_id = new_H_ids_from_gen_i[-1] + 1

                assert len(new_H_ids_from_gen_i) == r_H_id - l_H_id, 'new gen ' + \
                        'data must be in contiguous space'

                persis_info[i].update({'curr_H_ids': [l_H_id, r_H_id] })

    # Consensus step
    if num_gens_at_consensus == alloc_specs['user']['num_gens']:

        assert num_gens_at_consensus == len(avail_persis_worker_ids), \
                'All gens must be available, only {}/{} are though...'.format(
                    len(avail_worker_ids), len(num_gens_at_consensus))

        A = persis_info['alg_vars']['A']

        # compile location of x locations needed for Wx ...
        consensus_ids_in_H = np.array([persis_info[i]['curr_H_ids'][0] for i in avail_persis_worker_ids], dtype=int)
        num_gens = alloc_specs['user']['num_gens']
        n = len(gen_specs['user']['lb'])

        # TEMP
        c_step = persis_info.get('c_step', 0)
        if c_step % 3 ==1:
            gradg = np.zeros(num_gens*n, dtype=float)
            x     = np.zeros(num_gens*n, dtype=float)

        # send neighbors' {x_k underscore} between gens and prepare for new outer iter
        for i0, i in enumerate(avail_persis_worker_ids):
            
            incident_gens = A.indices[ A.indptr[i0]:A.indptr[i0+1] ]
            assert i0 not in incident_gens, 'no self loops permiited in ' + \
                    'adjacency matrix @A (i.e. only zeros on diagonal)'
            neighbor_consensus_ids_in_H = consensus_ids_in_H[ incident_gens ]

            # TEMP
            if c_step % 3 == 1:
                num_neighbors = len(neighbor_consensus_ids_in_H)
                x[i0*n:(i0+1)*n] = H[consensus_ids_in_H[i0]]['x']
                gradg[i0*n:(i0+1)*n] = num_neighbors*H[consensus_ids_in_H[i0]]['x'] \
                                    -np.sum(H[neighbor_consensus_ids_in_H]['x'], axis=0)

            gen_work(Work, i, ['x', 'gen_worker'], neighbor_consensus_ids_in_H,
                     persis_info.get(i), persistent=True)

            persis_info[i].update({'curr_H_ids': []})
            persis_info[i].update({'at_consensus': False})

        # TEMP
        if c_step % 3 == 1:
            R = persis_info['hyperparams']['R']
            print('g={:.4e}\n'.format(R * np.dot(x,gradg)), flush=True)

        persis_info.update({'c_step' : c_step+1})

        persis_info['outer_iter_ct'] += 1

    # partition sum of convex functions evenly (only do at beginning)
    if persis_info['outer_iter_ct'] == 1 and len( avail_worker_ids(W, persistent=False) ):
        num_funcs_arr = partition_funcs_arr(alloc_specs['user']['m'], 
                alloc_specs['user']['num_gens'])
        # k = persis_info['outer_iter_ct'] 
        A = persis_info['alg_vars']['A']
        S = get_doubly_stochastic(A)

    inactive_workers = np.sort(avail_worker_ids(W, persistent=False))
    for i0, i in enumerate(inactive_workers):

        # start up gens
        if persis_info['outer_iter_ct'] == 1 and gen_count < alloc_specs['user']['num_gens']:
            gen_count += 1
            l_idx = num_funcs_arr[gen_count-1]
            r_idx = num_funcs_arr[gen_count]

            S_i_indices = S.indices[S.indptr[i0]:S.indptr[i0+1]]
            S_i_gen_ids = inactive_workers[S_i_indices]     
            # gen S_i_gen_ids[i] corresponds to weight S_i_data[i]
            S_i_data = S.data[S.indptr[i0]:S.indptr[i0+1]] 

            persis_info[i].update({
                'f_i_idxs': range(l_idx, r_idx),
                'S_i_gen_ids': S_i_gen_ids,
                'S_i_data': S_i_data,
                'N':  persis_info['alg_vars']['N'],
                'eta': persis_info['alg_vars']['eta'],
                })
            persis_info[i].update({'at_consensus': False, 'curr_H_ids': []})

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

            gen_id = H[persis_info['next_to_give']]['gen_worker']
            [l_H_ids, r_H_ids] = persis_info[gen_id]['curr_H_ids']

            assert l_H_ids == persis_info['next_to_give'], \
                "@next_to_give={} does not match gen's requested work H id of {}".format(
                    persis_info['next_to_give'], l_H_ids)

            sim_work(Work, i, 
                     sim_specs['in'], 
                     np.arange(l_H_ids, r_H_ids),
                     persis_info.get(i))

            # we can safely assume the rows are contiguous due to !!
            persis_info['next_to_give'] += (r_H_ids - l_H_ids)

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

    L          = persis_info['hyperparams']['L']
    eps        = persis_info['hyperparams']['eps']
    N_const    = persis_info['hyperparams'].get('N_const', 1)
    step_const = persis_info['hyperparams'].get('step_const', 1)

    assert 0 < step_const <= 1, 'step scale is {:.4f} but must be in (0,1]'.format(step_const)

    # chain matrix
    num_gens = alloc_specs['user']['num_gens']
    diagonals = [np.ones(num_gens-1), np.ones(num_gens-1)]
    A = spp.csr_matrix( spp.diags(diagonals, [-1,1]) )
    S = get_doubly_stochastic(A)
    n = S.shape[0]
    rho = la.norm(S - (n**-1)*np.ones(S.shape), ord=2)

    assert rho < 1, 'Norm should be < 1, but is {:.4e}'.format(rho)

    eta = step_const * 1.0/L * min(1/6, (1-rho**2)**2/(4 * rho**2 * (3+4*rho**2) ))

    N = int(N_const / eps + 1)

    alg_vars = {
                'A': A,               # Adjacency matrix (we will not explicitly form Laplacian)
                'N': N,               # number of outer iterations 
                'eta': eta,           # step size
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
    n = A.shape[0]
    x = np.multiply( A.toarray() != 0,  np.random.random((n,n)))
    x = x + np.diag(np.random.random(n) + 1e-4)

    rsum = None
    csum = None

    while (np.any(rsum != 1)) | (np.any(csum != 1)):
        x = x / x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)

    x = spp.csr_matrix(x)

    return x
