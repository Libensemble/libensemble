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
    This allocation function will give simulation work if possible, but
    otherwise start up to one persistent generator. By default, evaluation
    results are given back to the generator once all generated points have
    been returned from the simulation evaluation. If alloc_specs['user']['async_return']
    is set to True, then any returned points are given back to the generator.

    If the single persistent generator has exited, then ensemble shutdown is triggered.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
        `test_persistent_surmise_calib.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_surmise_calib.py>`_ # noqa
    """

    Work = {}
    gen_count = count_persis_gens(W)

    # Exit if all persistent gens are done
    if (not persis_info.get('first_call', True)) and gen_count == 0:
        return Work, persis_info, 1

    for i in avail_worker_ids(W, persistent=True):

        # first check if gen has converged to a solution
        convg_sim_idxs_from_gen_i = np.where( 
                np.logical_and( H['converged'], H['gen_worker']==i )
                )[0]

        if len(convg_sim_idxs_from_gen_i):

            # assert i in persis_info['gen_list']

            # we should have only one convergence result, so access 0th elem
            idx = convg_sim_idxs_from_gen_i[0]
            convg_res = H[idx]
            persis_info[i]['num_f_evals'] = convg_res['num_f_evals']
            persis_info[i]['num_gradf_evals'] = convg_res['num_gradf_evals']
            if 'x_star' not in persis_info:
                persis_info['x_star'] = np.copy(convg_res['x'])
            else:
                x_i_idxs = double_extend(persis_info[i]['f_i_idxs'])
                persis_info['x_star'][x_i_idxs] = convg_res['x'][x_i_idxs]
            persis_info['num_convg_gens'] = 1 + persis_info.get('num_convg_gens', 0)
            H[idx]['converged'] = False

            if persis_info['num_convg_gens'] == len(persis_info['gen_list']):
                print('#########################')
                print('# FINAL RESULT ')
                print('#\n# x={}\n#'.format(persis_info['x_star']))
                for j in persis_info['gen_list']:
                    print('# gen {} had {} function and {} (full) gradient evals'.format(j, persis_info[j]['num_f_evals'], persis_info[j]['num_gradf_evals']))
                print('#')
                print('#########################')

                # TODO: can we return since immediately since everything has converged?
                # return Work, persis_info, 0

        ret_sim_idxs_from_gen_i = np.where( 
                np.logical_and(
                    H['returned'], np.logical_and(
                    ~H['ret_to_gen'], H['gen_worker']==i )
                ))[0]

        pt_ids_from_gen_i = set(H[ret_sim_idxs_from_gen_i]['pt_id'])

        if len(pt_ids_from_gen_i)==0:
            continue

        root_idxs = np.array([], dtype=int) # which history idxs have sum_i f_i
        f_i_idxs = persis_info[i].get('f_i_idxs')

        for pt_id in pt_ids_from_gen_i:

            subset_sim_idxs = np.where( H[ret_sim_idxs_from_gen_i]['pt_id'] == pt_id )[0]
            ret_sim_idxs_with_pt_id = ret_sim_idxs_from_gen_i[ subset_sim_idxs ]
            num_sims_req = H[ret_sim_idxs_with_pt_id][0]['num_sims_req']

            assert len(ret_sim_idxs_with_pt_id) <= num_sims_req, \
                    "{} incorrect number of sim data pts, expected <={}".format( 
                        len(ret_sim_idxs_with_pt_id), num_sims_req)

            # TODO: Find another away to determine when the work is all done 
            if len(ret_sim_idxs_with_pt_id) == num_sims_req:

                assert f_i_idxs is not None, print("gen worker does not have the required `f_i_idxs`")

                root_idxs = np.append(root_idxs, ret_sim_idxs_with_pt_id )
                H['ret_to_gen'][ ret_sim_idxs_with_pt_id ] = True # index by ['ret_to_gen'] first to avoid cpy

        if len(root_idxs) > 0:
            persis_info[i]['random']={'msg': 'dancing'}
            gen_work(Work, i, ['x', 'f_i', 'gradf_i'], np.atleast_1d(root_idxs), persis_info.get(i), persistent=True)

    num_req_gens = alloc_specs['user']['num_gens']

    # partition sum of convex functions evenly (only do at beginning)
    if persis_info.get('first_call', True) and len( avail_worker_ids(W, persistent=False) ):
        num_funcs_arr = partition_funcs_evenly_as_arr(alloc_specs['user']['num_gens'], num_req_gens)

    task_avail = ~H['given'] # & ~H['cancel_requested']

    for i in avail_worker_ids(W, persistent=False):

        # start up gens
        if persis_info.get('first_call', True) and gen_count < num_req_gens:

            gen_count += 1
            l_idx, r_idx = num_funcs_arr[gen_count-1], num_funcs_arr[gen_count]
            persis_info[i].update( {'f_i_idxs': range(l_idx, r_idx)} )

            # save gen ids to later access convergence results
            if 'gen_list' not in persis_info:
                persis_info['gen_list'] = [i]
            else:
                persis_info['gen_list'].append(i)

            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True)

        # give sim work when task available 
        elif np.any(task_avail):
            q_inds = 0 # start with oldest point in queue

            sim_ids_to_send = np.nonzero(task_avail)[0][q_inds]  
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info.get(i))

            task_avail[sim_ids_to_send] = False

        # this is awkward... no work todo... ¯\_(ツ)_/¯ ... yet!
        else:
            break

    if persis_info.get('first_call', True):
        persis_info['first_call'] = False

    return Work, persis_info, 0

def partition_funcs_evenly_as_arr(num_funcs, num_gens):
    num_funcs_arr = (num_funcs//num_gens) * np.ones(num_gens, dtype=int)
    num_leftover_funcs = num_funcs % num_gens
    num_funcs_arr[:num_leftover_funcs] += 1

    # builds starting and ending function indices for each gen e.g. if 7
    # functions split up amongst 3 gens, then num_funcs__arr = [0, 3, 5, 7]
    num_funcs_arr = np.append(0, np.cumsum(num_funcs_arr))

    return num_funcs_arr
