from libensemble.alloc_funcs.support import avail_worker_ids, sim_work, gen_work, test_any_gen

def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If all entries in `H` have been given a be evaluated, a worker
    is told to call the generator function, provided this wouldn't result in
    more than ``gen_specs['num_active_gen']`` active generators.

    :See:
        ``/libensemble/tests/regression_tests/test_fast_alloc.py``
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
