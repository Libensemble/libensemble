import numpy as np

from libensemble.alloc_funcs.support import avail_worker_ids, sim_work, gen_work, count_gens


def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most ``gen_specs['num_active_gens']``)
    generator instances.

    Allows for a ``gen_specs['batch_mode']`` where no generation
    work is given out unless all entries in ``H`` are returned.

    Allows for ``blocking`` of workers that are not active, for example, so
    their resources can be used for a different simulation evaluation.

    Can give points in highest priority, if ``'priority'`` is a field in ``H``.

    This is the default allocation function if one is not defined.

    :See:
        ``/libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``
    """

    Work = {}
    gen_count = count_gens(W)
    avail_set = set(W['worker_id'][np.logical_and(~W['blocked'],
                                                  W['active'] == 0)])

    for i in avail_worker_ids(W):

        if i not in avail_set:
            pass

        elif not np.all(H['allocated']):

            # Pick all high priority, oldest high priority, or just oldest point
            if 'priority' in H.dtype.fields:
                priorities = H['priority'][~H['allocated']]
                if gen_specs.get('give_all_with_same_priority'):
                    q_inds = (priorities == np.max(priorities))
                else:
                    q_inds = np.argmax(priorities)
            else:
                q_inds = 0

            # Get sim ids and check resources needed
            sim_ids_to_send = np.nonzero(~H['allocated'])[0][q_inds]
            sim_ids_to_send = np.atleast_1d(sim_ids_to_send)
            nodes_needed = (np.max(H[sim_ids_to_send]['num_nodes'])
                            if 'num_nodes' in H.dtype.names else 1)
            if nodes_needed > len(avail_set):
                break

            # Assign resources and mark tasks as allocated to workers
            sim_work(Work, i, sim_specs['in'], sim_ids_to_send, persis_info[i])
            H['allocated'][sim_ids_to_send] = True

            # Update resource records
            avail_set.remove(i)
            if nodes_needed > 1:
                workers_to_block = list(avail_set)[:nodes_needed-1]
                avail_set.difference_update(workers_to_block)
                Work[i]['libE_info']['blocking'] = workers_to_block

        else:

            # Allow at most num_active_gens active generator instances
            if gen_count >= gen_specs.get('num_active_gens', gen_count+1):
                break

            # No gen instances in batch mode if workers still working
            still_working = ~H['returned']
            if gen_specs.get('batch_mode') and np.any(still_working):
                break

            # Give gen work
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info[i])

    return Work, persis_info
