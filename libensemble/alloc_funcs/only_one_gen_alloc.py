from libensemble.tools.alloc_support import avail_worker_ids, sim_work, gen_work, test_any_gen


def ensure_one_active_gen(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If there is no active generator, then one is started.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    Work = {}
    gen_flag = True

    for i in avail_worker_ids(W):
        if persis_info['next_to_give'] < len(H):

            # Give sim work if possible
            sim_work(Work, i, sim_specs['in'], [persis_info['next_to_give']], [])
            persis_info['next_to_give'] += 1

        elif not test_any_gen(W) and gen_flag:

            if not all(H['returned']):
                break

            # Give gen work
            persis_info['total_gen_calls'] += 1
            gen_flag = False
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info[i])

    return Work, persis_info
