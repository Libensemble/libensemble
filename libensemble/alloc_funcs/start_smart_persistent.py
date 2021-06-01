import numpy as np
from libensemble.tools.alloc_support import (avail_worker_ids, sim_work, gen_work,
                                             count_persis_gens, all_returned)


def only_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
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

    # Step I: Setup for first call
    if persis_info.get('first_call', True):
        # TODO: Can build onto nested dictionary
        assert np.all(H['given']), "Initial points in H have never been given."
        assert np.all(H['given_back']), "Initial points in H have never been given_back."
        assert all_returned(H), "Initial points in H have never been returned."
        persis_info['fields_to_give_back'] = ['f'] + [n[0] for n in gen_specs['out']]

        # persis_info['samples_in_H0'] = sum(H['local_pt'] == 0)
        persis_info['next_to_give'] = len(H)  #
        persis_info['first_call'] = False

    # Exit if all persistent gens are done
    elif gen_count == 0:
        return Work, persis_info, 1

    # Step Ia: check for new points
    if len(H) != persis_info['H_len']:
        # Something new is in the history.
        persis_info['need_to_give'].update(H['sim_id'][persis_info['H_len']:].tolist())
        persis_info['H_len'] = len(H)
        persis_info['pt_ids'] = set(np.unique(H['pt_id']))
        for pt_id in persis_info['pt_ids']:
            persis_info['inds_of_pt_ids'][pt_id] = H['pt_id'] == pt_id

    # Step II: When need to assign new points , prune out bad ones, e.g.
    # f_1(x_j) > prev_min
    pt_ids_to_pause = set() 
    if len(persis_info['need_to_give']) > 0 and \
        alloc_specs['user'].get('stop_partial_eval'):

        pt_ids = set(persis_info['pt_ids']) - persis_info['complete']     # set difference
        pt_ids = np.array(list(pt_ids))
        partial_fvals = np.zeros(len(pt_ids))

        for j, pt_id in enumerate(pt_ids):

            # TODO: can we track which sim worker was responsible w/o this massive array?
            a1 = persis_info['inds_of_pt_ids'][pt_id]
    
            # TODO: Store f(x) = \sum\limits_i f_i(x)
            if np.all(H['returned'][a1]): 
                persis_info['complete'].add(pt_id)
                values = gen_specs['user']['combine_component_func'](H['f_i'][a1])
                persis_info['best_complete_val'] = min(persis_info['best_complete_val'], values)
            else:
                partial_fvals[j] = gen_specs['user']['combine_component_func'](H['f_i'][a1])
    
        if len(persis_info['complete']) and len(pt_ids) > 1:
            worse_flag = np.zeros(len(pt_ids), dtype=bool)

            for j, pt_id in enumerate((pt_ids)):

                if (pt_id not in persis_info['complete']) and \
                    (partial_fvals[j] > 0.1*persis_info['best_complete_val']):

                    print("found better!!", flush=True)

                    pt_ids_to_pause.update({pt_id})

        new_pts_to_remove = not pt_ids_to_pause.issubset(persis_info['already_paused'])
        if new_pts_to_remove:
            persis_info['already_paused'].update(pt_ids_to_pause)

            sim_ids_to_remove = np.in1d(H['pt_id'], list(pt_ids_to_pause)) 
            H['paused'][sim_ids_to_remove] = True

            persis_info['need_to_give'] = persis_info['need_to_give'] - set(np.where(sim_ids_to_remove)[0])  

    # Step III: Give complete pts back to gen
    active_recv_gen = False # TEMP
    for i in avail_worker_ids(W, persistent=True):
        # NOTE: Also check alloc_func/persistent_aposmm_alloc.py

        gen_inds = (H['gen_worker'] == i)
        returned_but_not_given = np.logical_and.reduce((H['returned'], ~H['given_back'], gen_inds))
        # TODO: allow return of incomplete partial evalutions

        # If all pts returned by sim but not given back to gen, then give back to gen
        if np.all(returned_but_not_given):
            inds_since_last_gen = np.where(returned_but_not_given)[0]
            gen_work(Work, i,
                     ['f','f_i','obj_component','sim_id'], # components to send back
                     np.atleast_1d(inds_since_last_gen), persis_info.get(i), persistent=True,
                     active_recv=active_recv_gen)

    task_avail = ~H['given'] & ~H['cancel_requested']

    # Step IV: Generate sim/gen work
    for i in avail_worker_ids(W, persistent=False):

        # Start up number of requested gens
        if gen_count < alloc_specs['user']['num_gens']:

            # Finally, call a persistent generator as there is nothing else to do.
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True, active_recv=active_recv_gen)

        # Once all gens running, give sim work when task available (i.e. data is given and ready-to-go)
        elif np.any(task_avail):
            # Ignore priority for now
            # if 'priority' in H.dtype.fields:
            #     priorities = H['priority'][task_avail]
            #     if gen_specs['user'].get('give_all_with_same_priority'):
            #         q_inds = (priorities == np.max(priorities))
            #    else:
            #        q_inds = np.argmax(priorities)
            # else:
            #     q_inds = 0
            q_inds = 0

            # perform sim evaluations (if they exist in History).
            sim_ids_to_send = np.nonzero(task_avail)[0][q_inds]  # oldest point(s)
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info.get(i))
            task_avail[sim_ids_to_send] = False

        # nothing else to do, exit for loop until new work
        else:
            break

    return Work, persis_info, 0
