import numpy as np
import numpy.linalg as la
from libensemble.tools.alloc_support import (avail_worker_ids, sim_work, gen_work,
                                             count_persis_gens, all_returned)

# TODO: Place this in support file
def double_extend(arr):
    out = np.zeros(len(arr)*2, dtype=type(arr[0]))
    out[0::2] = 2*np.array(arr)
    out[1::2] = 2*np.array(arr)+1
    return out

def start_persistent_independent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This alloc function solves a sum of functions, specifically, one where
    $f_i$ is indpendent from $f_j$ from $i \ne j$.

    During the development of this code, we realized indexing the entire H
    was a signficant bottleneck on the alloc (moreso than the gen and sim).
    Therefore, we include persis_info['last_Hlen'] so that

        H[ persis_info['last_Hlen'] : len(H) ]

    contains new inputs (queried by gens). For when a gen sent data and is
    waiting on sim to return the results, we also include 
    ```persis_info.get(i)['queued_H_ids']```, which are indexes of H that
    must be completed (i.e. H['returned'] = True) for the alloc to return to
    the gen.
    """

    Work = {}
    curr_num_gens = count_persis_gens(W)

    # Exit if all persistent gens are done
    if (not persis_info.get('first_call', True)) and curr_num_gens == 0:
        return Work, persis_info, 1

    for i in avail_worker_ids(W, persistent=True):

        # if has already converged, skip
        if i in persis_info.get('convg_gens', []):
            continue

        # Is gen waiting on work to be completed?
        if len(persis_info[i].get('curr_H_ids', [])):

            [l_H_id, r_H_id] = persis_info[i].get('curr_H_ids')
            num_sims_req_by_gen = r_H_id - l_H_id

            num_fin_sims = np.sum(H['returned'][l_H_id:r_H_id])
            # fin_sims_for_gen_i_subarr = np.where( 
            #         np.logical_and(
            #             H[l_H_id:r_H_id]['returned'], 
            #             ~H[l_H_id:r_H_id]['ret_to_gen']
            #         ))[0]

            # if len(fin_sims_for_gen_i_subarr) == (r_H_id - l_H_id):
            if num_fin_sims == num_sims_req_by_gen:

                sims_to_ret_to_gen = np.arange(l_H_id, r_H_id)

                gen_work(Work, i, 
                         ['x', 'f_i', 'gradf_i'], 
                         sims_to_ret_to_gen,
                         persis_info.get(i), 
                         persistent=True)

                # index by ['ret_to_gen'] first avoid writing to cpy
                # H['ret_to_gen'][fin_sims_for_gen_i] = True 
                persis_info[i].update({'curr_H_ids': []})

        else:
            last_H_len = persis_info['last_H_len']

            # first check if gen has converged to a solution
            convg_sim_ids = np.where( 
                np.logical_and( 
                    H[last_H_len:]['converged'], 
                    H[last_H_len:]['gen_worker']==i )
                )[0] 

            if len(convg_sim_ids):

                convg_sim_ids += last_H_len # re-orient

                # assert i in persis_info['gen_list']

                # we should have only one convergence result, so access 0th elem
                assert len(convg_sim_ids) == 1, print('should only have one \
                        convergence result, but received {}'.format(len(convg_sim_ids)))

                convg_gens = persis_info.get('convg_gens', [])
                convg_gens.append(i)
                persis_info.update({'convg_gens': convg_gens})

                idx = convg_sim_ids[0]
                convg_res = H[idx]
                persis_info[i]['num_f_evals'] = convg_res['num_f_evals']
                persis_info[i]['num_gradf_evals'] = convg_res['num_gradf_evals']
                if 'x_star' not in persis_info:
                    persis_info['x_star'] = np.copy(convg_res['x'])
                else:
                    x_i_idxs = double_extend(persis_info[i]['f_i_idxs'])
                    persis_info['x_star'][x_i_idxs] = convg_res['x'][x_i_idxs]
                # TODO: relieve the gens since they are waiting for communication

                if len(persis_info.get('convg_gens', [])) == len(persis_info['gen_list']):
                    print('#########################')
                    print('# FINAL RESULT ')
                    x_star = persis_info['x_star']
                    print('#\n# x={}'.format(x_star))
                    n = len(x_star)
                    print('#|x_final - 1_n|_2/|1_n|_2 = {:.4f}\n'.format(
                        la.norm(np.ones(n) - x_star)/n**0.5))
                    for j in persis_info['gen_list']:
                        print('# gen {} had {} function and {} (full) gradient evals'.format(j, persis_info[j]['num_f_evals'], persis_info[j]['num_gradf_evals']))
                    print('#')
                    print('#########################')

                    # TODO: can we return since immediately since everything has converged?
                    return Work, persis_info, 1

            # otherwise, if gen must have requested new sim work
            else:
                new_H_ids_from_gen_i = np.where( H[last_H_len:]['gen_worker'] == i )[0]

                if len(new_H_ids_from_gen_i) == 0:
                    import ipdb; ipdb.set_trace()
                assert len(new_H_ids_from_gen_i), print("Gen must request new sim work or show convergence if avail, but neither occured")

                # re-orient
                new_H_ids_from_gen_i += last_H_len

                l_H_id = new_H_ids_from_gen_i[0]
                r_H_id = new_H_ids_from_gen_i[-1] + 1

                assert len(new_H_ids_from_gen_i) == r_H_id - l_H_id, print("new gen data must be in contiguous space")

                persis_info[i].update({'curr_H_ids': [l_H_id, r_H_id] })

    num_req_gens = alloc_specs['user']['num_gens']
    m = gen_specs['user']['m']

    # partition sum of convex functions evenly (only do at beginning)
    if persis_info.get('first_call', True) and len( avail_worker_ids(W, persistent=False) ):
        num_funcs_arr = partition_funcs_evenly_as_arr(m, num_req_gens)

    for i in avail_worker_ids(W, persistent=False):

        # start up gens
        if persis_info.get('first_call', True) and curr_num_gens < num_req_gens:

            curr_num_gens += 1
            l_idx, r_idx = num_funcs_arr[curr_num_gens-1], num_funcs_arr[curr_num_gens]
            persis_info[i].update( {'f_i_idxs': range(l_idx, r_idx)} )

            # save gen ids to later access convergence results
            if 'gen_list' not in persis_info:
                persis_info['gen_list'] = [i]
            else:
                persis_info['gen_list'].append(i)

            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True)

        # give sim work when task available 
        elif persis_info['next_to_give'] < len(H):
            while persis_info['next_to_give'] < len(H) and \
                  (H[persis_info['next_to_give']]['given'] or \
                   H[persis_info['next_to_give']]['converged']):

                persis_info['next_to_give'] += 1

            if persis_info['next_to_give'] >= len(H):
                break

            sim_work(Work, i, 
                     sim_specs['in'], 
                     np.array([persis_info['next_to_give']]), 
                     persis_info.get(i))

            persis_info['next_to_give'] += 1

        # this is awkward... no work todo... ¯\_(ツ)_/¯ ... yet!
        else:
            break

    if persis_info.get('first_call', True):
        persis_info['first_call'] = False

    persis_info.update({'last_H_len' : len(H)})

    return Work, persis_info, 0

def partition_funcs_evenly_as_arr(num_funcs, num_gens):
    num_funcs_arr = (num_funcs//num_gens) * np.ones(num_gens, dtype=int)
    num_leftover_funcs = num_funcs % num_gens
    num_funcs_arr[:num_leftover_funcs] += 1

    # builds starting and ending function indices for each gen e.g. if 7
    # functions split up amongst 3 gens, then num_funcs__arr = [0, 3, 5, 7]
    num_funcs_arr = np.append(0, np.cumsum(num_funcs_arr))

    return num_funcs_arr
