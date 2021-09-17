import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def only_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to ``alloc_specs['user']['num_active_gens']``
    persistent generators (defaulting to one).

    By default, evaluation results are given back to the generator once
    all generated points have been returned from the simulation evaluation.
    If ``alloc_specs['user']['async_return']`` is set to True, then any
    returned points are given back to the generator.

    If any of the persistent generators has exited, then ensemble shutdown
    is triggered.

    **User options**:

    To be provided in calling script: E.g., ``alloc_specs['user']['async_return'] = True``

    init_sample_size: int, optional
        Initial sample size - always return in batch. Default: 0

    num_active_gens: int, optional
        Maximum number of persistent generators to start. Default: 1

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
        `test_persistent_uniform_gen_decides_stop.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_gen_decides_stop.py>`_ # noqa
    """

    # Initialize alloc_specs['user'] as user.
    user = alloc_specs.get('user', {})
    sched_opts = user.get('scheduler_opts', {})
    active_recv_gen = user.get('active_recv_gen', False)  # Persistent gen can handle irregular communications
    init_sample_size = user.get('init_sample_size', 0)   # Always batch return until this many evals complete
    batch_give = gen_specs['user'].get('give_all_with_same_priority', False)  # SH TODO: alloc_specs or gen_specs?

    support = AllocSupport(W, H, persis_info, sched_opts)
    gen_count = support.count_persis_gens()
    Work = {}

    # Asynchronous return to generator
    async_return = user.get('async_return', False) and sum(H['returned']) >= init_sample_size

    # gen_specs['persis_in']
    gen_return_fields = sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')]

    if gen_count < persis_info.get('num_gens_started', 0):
        # When a persistent worker is done, trigger a shutdown (returning exit condition of 1)
        return Work, persis_info, 1

    # Give evaluated results back to a running persistent gen
    for wid in support.avail_worker_ids(persistent=EVAL_GEN_TAG, active_recv=active_recv_gen):
        gen_inds = (H['gen_worker'] == wid)
        returned_but_not_given = np.logical_and.reduce((H['returned'], ~H['given_back'], gen_inds))
        if np.any(returned_but_not_given):
            if async_return or support.all_returned(H, gen_inds):
                point_ids = np.where(returned_but_not_given)[0]
                Work[wid] = support.gen_work(wid, gen_return_fields, point_ids, persis_info.get(wid),
                                 persistent=True, active_recv=active_recv_gen)
                returned_but_not_given[point_ids] = False

    # SH TODO: Now the give_sim_work_first bit
    points_to_evaluate = ~H['given'] & ~H['cancel_requested']
    avail_workers = support.avail_worker_ids(persistent=False, zero_resource_workers=False)
    for wid in avail_workers:

        if not np.any(points_to_evaluate):
            break

        sim_ids_to_send = support.points_by_priority(H, points_avail=points_to_evaluate, batch=batch_give)
        try:
            Work[wid] = support.sim_work(wid, H, sim_specs['in'], sim_ids_to_send, persis_info.get(wid))
        except InsufficientFreeResources:
            break

        points_to_evaluate[sim_ids_to_send] = False

    # Start persistent gens if no worker to give out. Uses zero_resource_workers if defined.
    if not np.any(points_to_evaluate):
        avail_workers = support.avail_worker_ids(persistent=False, zero_resource_workers=True)

        for wid in avail_workers:
            if gen_count < user.get('num_active_gens', 1):
                # Finally, start a persistent generator as there is nothing else to do.
                try:
                    Work[wid] = support.gen_work(wid, gen_specs['in'], range(len(H)), persis_info.get(wid),
                                     persistent=True, active_recv=active_recv_gen)
                except InsufficientFreeResources:
                    break
                persis_info['num_gens_started'] = persis_info.get('num_gens_started', 0) + 1
                gen_count += 1

    del support
    return Work, persis_info, 0
