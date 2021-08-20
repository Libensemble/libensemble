import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


# SH TODO: Either replace only_persistent_gens or add a different alloc func (or file?)
#          Check/update docstring

# SH TODO: New support funcs terminology.
#          H boolean filters?  e.g. [True, False, False] e.g. prefix H_filter or Hmask_
#          e.g. mask_pts_to_eval =  support.getmask_points_to_evaluate()

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

    # Initialize alloc_specs['user'] as user.
    user = alloc_specs.get('user', {})
    sched_opts = user.get('scheduler_opts', {})
    active_recv_gen = user.get('active_recv_gen', False)  # Persistent gen can handle irregular communications
    init_sample_size = user.get('init_sample_size', 0)   # Always batch return until this many evals complete
    batch_give = gen_specs['user'].get('give_all_with_same_priority', False)  # SH TODO: Should this be alloc_specs not gen_specs?

    support = AllocSupport(W, H, persis_info, sched_opts)
    gen_count = support.count_persis_gens()
    Work = {}

    # Asynchronous return to generator
    async_return = user.get('async_return', False) and sum(H['returned']) >= init_sample_size

    # gen_specs['persis_in']
    gen_return_fields = sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')]

    # SH TODO: Generalize this
    if persis_info.get('gen_started') and gen_count == 0:
        # The one persistent worker is done. Exiting
        return Work, persis_info, 1

    # Give evaluated results back to a running persistent gen
    for wid in support.avail_worker_ids(persistent=EVAL_GEN_TAG, active_recv=active_recv_gen):
        # SH TODO: points_evaluated terminology? Also its an H boolean filter (maybe filter_points_evaluated?)
        points_evaluated = support.get_evaluated_points(gen=wid)
        if np.any(points_evaluated):
            if async_return or support.all_returned(gen=wid):
                point_ids = np.where(points_evaluated)[0]
                support.gen_work(Work, wid, gen_return_fields, point_ids, persis_info.get(wid),
                                 persistent=True, active_recv=active_recv_gen)
                # SH TODO: This should use filter like points_to_evaluate below... 'given_back' to be libE field set by manager.
                #          Could be set in libE_info here - and manager reads that? Or manager could determine existing point.
                #          But as read in get_evaluated_points, must be written somewhere - cant remove this till manager does it.
                #H['given_back'][point_ids] = True  # SH TODO: Move to manager (libE field) when give to gen points with 'returned' True
                points_evaluated[point_ids] = False  # SH TODO: May not need if each iteration is mutually exclusive points!

    # SH TODO: Now the give_sim_work_first bit
    points_to_evaluate = support.get_points_to_evaluate()  # SH TODO: Again an H boolean filter
    avail_workers = support.avail_worker_ids(persistent=False, zero_resource_workers=False)
    for wid in avail_workers:

        if not np.any(points_to_evaluate):
            break

        sim_ids_to_send = support.points_by_priority(points_avail=points_to_evaluate, batch=batch_give)
        try:
            support.sim_work(Work, wid, sim_specs['in'], sim_ids_to_send, persis_info.get(wid))
        except InsufficientFreeResources:
            break

        points_to_evaluate[sim_ids_to_send] = False

    # A separate loop/section as now need zero_resource_workers for gen.
    if not np.any(points_to_evaluate):
        avail_workers = support.avail_worker_ids(persistent=False, zero_resource_workers=True)

        # SH TODO: So we don't really need a loop here for this, but general case would allow multiple gens
        for wid in avail_workers:
            if gen_count == 0:
                # Finally, call a persistent generator as there is nothing else to do.
                gen_count += 1
                try:
                    support.gen_work(Work, wid, gen_specs['in'], range(len(H)), persis_info.get(wid),
                                     persistent=True, active_recv=active_recv_gen)
                except InsufficientFreeResources:
                    break
                persis_info['gen_started'] = True

    return Work, persis_info, 0
