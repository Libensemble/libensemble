import numpy as np
import numpy.linalg as la
from libensemble.tools.alloc_support import (avail_worker_ids, sim_work, gen_work,
                                             count_persis_gens, all_returned)


def start_smart_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
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

    if persis_info.get('first_call', True):
        persis_info['first_call'] = False

    # Exit if all persistent gens are done
    elif gen_count == 0:
        return Work, persis_info, 1

    m = gen_specs['user']['m']

    for i in avail_worker_ids(W, persistent=True):

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

            # if len(ret_sim_idxs_with_pt_id) > num_sims_req:
            #     import ipdb; ipdb.set_trace()
            #     uuu = 1

            assert len(ret_sim_idxs_with_pt_id) <= num_sims_req, \
                    "{} incorrect number of sim data pts, expected <={}".format( 
                        len(ret_sim_idxs_with_pt_id), num_sims_req)

            # TODO: Find another away to determine when the work is all done 
            if len(ret_sim_idxs_with_pt_id) == num_sims_req:
                # No summation since distributed optimization solves each f_i with own x_i

                # returned_fvals = H[ ret_sim_idxs_with_pt_id ]['gradf_i']
                # grad_f = np.sum( returned_fvals, axis=0 )
                # print("Norm: {:.4f} [{}]".format(la.norm(grad_f), len(grad_f)), flush=True)

                assert f_i_idxs is not None, print("gen worker does not have the required `f_i_idxs`")

                root_idxs = np.append(root_idxs, ret_sim_idxs_with_pt_id )
                H['ret_to_gen'][ ret_sim_idxs_with_pt_id ] = True # index by ['ret_to_gen'] first to avoid cpy

        if len(root_idxs) > 0:
            gen_work(Work, i, ['x', 'f_i', 'gradf_i'], np.atleast_1d(root_idxs), persis_info.get(i), persistent=True)

    task_avail = ~H['given'] # & ~H['cancel_requested']

    num_gen_workers = alloc_specs['user']['num_gens']

    # partition sum of convex functions evenly (only do at beginning)
    if not persis_info.get('init_gens', False) and len( avail_worker_ids(W, persistent=False) ):

        num_funcs_arr = (m//num_gen_workers) * np.ones(num_gen_workers, dtype=int)
        num_leftover_funcs = m % num_gen_workers
        num_funcs_arr[:num_leftover_funcs] += 1
        # builds starting and ending function indices for each gen e.g. if 7
        # functions split up amongst 3 gens, then num_funcs__arr = [0, 3, 5, 7]
        num_funcs_arr = np.append(0, np.cumsum(num_funcs_arr))

    for i in avail_worker_ids(W, persistent=False):

        # start up gens
        if not persis_info.get('init_gens', False) and gen_count < num_gen_workers:

            gen_count += 1
            l_idx, r_idx = num_funcs_arr[gen_count-1], num_funcs_arr[gen_count]
            persis_info[i].update( {'f_i_idxs': range(l_idx, r_idx)} )

            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True)

            if gen_count == num_gen_workers:
                persis_info['init_gens'] = True

        # give sim work when task available 
        elif np.any(task_avail):
            q_inds = 0 # start with oldest point in queue

            sim_ids_to_send = np.nonzero(task_avail)[0][q_inds]  
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info.get(i))

            task_avail[sim_ids_to_send] = False

        else:
            break

    return Work, persis_info, 0
