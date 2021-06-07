from libensemble.tools.alloc_support import avail_worker_ids, sim_work


def give_pregenerated_sim_work(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in alloc_spec['x'] to
    idle workers. It is an example use case where no gen_func is used.

    .. seealso::
        `test_fast_alloc.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_fast_alloc.py>`_ # noqa
    """

    Work = {}
    # Unless already defined, initialize next_to_give to be the first point in H
    persis_info['next_to_give'] = persis_info.get('next_to_give', 0)

    if persis_info['next_to_give'] >= len(H):
        return Work, persis_info, 1

    for i in avail_worker_ids(W):
        # Skip any cancelled points
        while persis_info['next_to_give'] < len(H) and H[persis_info['next_to_give']]['cancel_requested']:
            persis_info['next_to_give'] += 1

        # Give sim work
        sim_work(Work, i, sim_specs['in'], [persis_info['next_to_give']], [])
        persis_info['next_to_give'] += 1

        if persis_info['next_to_give'] >= len(H):
            break

    return Work, persis_info
