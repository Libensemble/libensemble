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

    **User options**:

    To be provided in calling script: E.g., ``alloc_specs['user']['async_return'] = True``

    init_sample_size: int, optional
        Initial sample size - always return in batch. Default: 0

    async_return: boolean, optional
        Return results to gen as they come in (after sample). Default: False (batch return).

    active_recv_gen: boolean, optional
        Create gen in active receive mode. If True, the manager does not need to wait
        for a return from the generator before sending further returned points.
        Default: False


    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
        `test_persistent_surmise_calib.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_surmise_calib.py>`_ # noqa
    """

    Work = {}
    gen_count = count_persis_gens(W)

    import ipdb; ipdb.set_trace(context=5)

    # Setup for first call
    if persis_info.get('first_call', True):
        assert np.all(H['given']), "Initial points in H have never been given."
        assert np.all(H['given_back']), "Initial points in H have never been given_back."
        assert all_returned(H), "Initial points in H have never been returned."
        persis_info['fields_to_give_back'] = ['f'] + [n[0] for n in gen_specs['out']]

        # persis_info['samples_in_H0'] = sum(H['local_pt'] == 0)
        persis_info['next_to_give'] = len(H)  #
        persis_info['first_call'] = False

    elif gen_count == 0:
        # Exit once all persistent gens are done
        return Work, persis_info, 1

    # Initialize alloc_specs['user'] as user.
    user = alloc_specs.get('user', {})
    active_recv_gen = user.get('active_recv_gen', False) # Persistent gen can handle irregular communications
    init_sample_size = user.get('init_sample_size', 0)   # Always batch return until this many evals complete

    # Asynchronous return to generator
    async_return = user.get('async_return', False) and sum(H['returned']) >= init_sample_size

    # Give evaluated results back to a running persistent gen
    # for i in avail_worker_ids(W, persistent=True):
    # TODO: Figure out what active_recv_gen is
    # NOTE: Also check alloc_func/persistent_aposmm_alloc.py
    for i in avail_worker_ids(W, persistent=True, active_recv=active_recv_gen):
        gen_inds = (H['gen_worker'] == i)
        returned_but_not_given = np.logical_and.reduce((H['returned'], ~H['given_back'], gen_inds))
        if np.any(returned_but_not_given):
            inds_since_last_gen = np.where(returned_but_not_given)[0]
            # TODO: Check to make it worthwhile to send to gen
            gen_work(Work, i,
                     ['f','f_i','obj_component','sim_id'], # what components to send back
                     np.atleast_1d(inds_since_last_gen), persis_info.get(i), persistent=True,
                     active_recv=active_recv_gen)

    # TODO: Prune out points that are prematurely bad

    task_avail = ~H['given'] & ~H['cancel_requested']
    num_req_gens = alloc_specs['user']['num_gens']
    for i in avail_worker_ids(W, persistent=False):

        # Start up number of requested gens
        if gen_count < num_req_gens:

            # Finally, call a persistent generator as there is nothing else to do.
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True, active_recv=active_recv_gen)

        # Once all gens running, give sim work when task available (i.e. data is given and read-to-go)
        elif np.any(task_avail):
            if 'priority' in H.dtype.fields:
                priorities = H['priority'][task_avail]
                if gen_specs['user'].get('give_all_with_same_priority'):
                    q_inds = (priorities == np.max(priorities))
                else:
                    q_inds = np.argmax(priorities)
            else:
                q_inds = 0

            # perform sim evaluations (if they exist in History).
            sim_ids_to_send = np.nonzero(task_avail)[0][q_inds]  # oldest point(s)
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info.get(i))
            task_avail[sim_ids_to_send] = False

        # nothing else to do, exit for loop until new work
        else:
            break

    return Work, persis_info, 0
