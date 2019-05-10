from libensemble.alloc_funcs.support import avail_worker_ids, sim_work


def give_pregenerated_sim_work(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in alloc_spec['x'] to
    idle workers. It is an example use case where no gen_func is used.

    :See:
        ``/libensemble/tests/regression_tests/test_fast_alloc.py``
    """

    Work = {}
    if not persis_info:
        persis_info['next_to_give'] = 0

    assert persis_info['next_to_give'] < len(H), 'No more work to give inside give_pregenerated_sim_work.'

    for i in avail_worker_ids(W):
        # Give sim work
        sim_work(Work, i, sim_specs['in'], [persis_info['next_to_give']], [])
        persis_info['next_to_give'] += 1

    return Work, persis_info
