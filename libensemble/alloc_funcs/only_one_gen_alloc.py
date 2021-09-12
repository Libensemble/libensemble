from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


def ensure_one_active_gen(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If there is no active generator, then one is started.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    user = alloc_specs.get('user', {})
    sched_opts = user.get('scheduler_opts', {})
    support = AllocSupport(W, H, persis_info, sched_opts)

    Work = {}
    gen_flag = True
    gen_in = gen_specs.get('in', [])

    for wid in support.avail_worker_ids():

        # Skip any cancelled points
        while persis_info['next_to_give'] < len(H) and H[persis_info['next_to_give']]['cancel_requested']:
            persis_info['next_to_give'] += 1

        if persis_info['next_to_give'] < len(H):
            try:
                support.sim_work(Work, wid, sim_specs['in'], [persis_info['next_to_give']], [])
            except InsufficientFreeResources:
                break
            persis_info['next_to_give'] += 1

        elif not support.test_any_gen() and gen_flag:

            if not support.all_returned():
                break

            # Give gen work
            return_rows = range(len(H)) if gen_in else []
            try:
                support.gen_work(Work, wid, gen_in, return_rows, persis_info.get(wid))
            except InsufficientFreeResources:
                break
            gen_flag = False
            persis_info['total_gen_calls'] += 1

    del support
    return Work, persis_info
