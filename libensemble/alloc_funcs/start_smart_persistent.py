import numpy as np
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
                np.logical_and(H['returned'], 
                    np.logical_and(~H['ret_to_gen'], H['gen_worker']==i ))
                )[0]

        pt_ids_from_gen_i = set(H[ret_sim_idxs_from_gen_i]['pt_id'])

        if len(pt_ids_from_gen_i)==0:
            continue

        root_idxs = np.array([], dtype=int) # which history idxs have sum_i f_i

        for pt_id in pt_ids_from_gen_i:

            # TODO: Can we do consecutive accesses, e.g. tuple (10,44) vs arr [10,11,...,44]?
            subset_sim_idxs = np.where( H[ret_sim_idxs_from_gen_i]['pt_id'] == pt_id )[0]
            ret_sim_idxs_with_pt_id = ret_sim_idxs_from_gen_i[ subset_sim_idxs ]

            assert len(ret_sim_idxs_with_pt_id) <= m, \
                    "{} incorrect number of sim data pts, expected {}".format( 
                        len(returned_pt_id_sim_idxs), m)

            if len(ret_sim_idxs_with_pt_id) == m:

                # root_idx = ret_sim_idxs_with_pt_id[0]

                # store the sum of {f_i}'s into the first idx (to reduce comm)
                # returned_fvals = H[ ret_sim_idxs_with_pt_id ]['f_i']
                # H[ root_idx ]['f_i'] = gen_specs['user']['combine_component_func'](returned_fvals)

                returned_fvals = H[ ret_sim_idxs_with_pt_id ]['gradf_i']

                # H[ root_idx ]['gradf_i'] = gen_specs['user']['combine_component_func'](returned_fvals)
                # root_idxs.append(root_idx)

                grad_f = gen_specs['user']['combine_component_func'](returned_fvals)
                print("Gradient: {}".format(grad_f), flush=True)

                root_idxs = np.append(root_idxs, ret_sim_idxs_with_pt_id )
                H['ret_to_gen'][ ret_sim_idxs_with_pt_id ] = True # accessing ['ret_to_gen'] is to ensure we do not write to cpy

        if len(root_idxs) > 0:
            gen_work(Work, i, ['x', 'gradf_i'], np.atleast_1d(root_idxs), persis_info.get(i), persistent=True)

    task_avail = ~H['given'] # & ~H['cancel_requested']

    for i in avail_worker_ids(W, persistent=False):

        # start up gens
        if gen_count < alloc_specs['user']['num_gens']:

            # Finally, call a persistent generator as there is nothing else to do.
            gen_count += 1
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
