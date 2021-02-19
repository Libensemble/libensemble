import numpy as np
# SH TODO:  Consider importing a class and using as object functions
from libensemble.tools.alloc_support import (sim_work, gen_work, count_gens,
                                             get_avail_workers_by_group, assign_workers)


# SH TODO: Either replace give_sim_work_first or add a different alloc func (or file?)
#          Check/update docstring
def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most ``alloc_specs['user']['num_active_gens']``)
    generator instances.

    Allows for a ``alloc_specs['user']['batch_mode']`` where no generation
    work is given out unless all entries in ``H`` are returned.

    Allows for ``blocking`` of workers that are not active, for example, so
    their resources can be used for a different simulation evaluation.

    Can give points in highest priority, if ``'priority'`` is a field in ``H``.

    This is the default allocation function if one is not defined.

    .. seealso::
        `test_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling.py>`_ # noqa
    """

    Work = {}
    gen_count = count_gens(W)

    # Dictionary of workers by group (node).
    avail_workers = get_avail_workers_by_group(W)
    # print('avail_workers is', avail_workers)

    while any(avail_workers.values()):

        if not np.all(H['allocated']):

            # Pick all high priority, oldest high priority, or just oldest point
            if 'priority' in H.dtype.fields:
                priorities = H['priority'][~H['allocated']]
                if gen_specs['user'].get('give_all_with_same_priority'):
                    q_inds = (priorities == np.max(priorities))
                else:
                    q_inds = np.argmax(priorities)
            else:
                q_inds = 0

            # Get sim ids and check resources needed
            sim_ids_to_send = np.nonzero(~H['allocated'])[0][q_inds]
            sim_ids_to_send = np.atleast_1d(sim_ids_to_send)

            nworkers_req = (np.max(H[sim_ids_to_send]['resource_sets'])
                            if 'resource_sets' in H.dtype.names else 1)

            # If more than one group (node) required, allocates whole nodes - also removes from avail_workers
            worker_team = assign_workers(avail_workers, nworkers_req)
            # print('AFTER ASSIGN sim ({}): avail_workers: {}'.format(worker_team,avail_workers),flush=True)

            if not worker_team:
                break  # No slot found - insufficient available resources for this work unit
            worker = worker_team[0]

            sim_work(Work, worker, sim_specs['in'], sim_ids_to_send, persis_info[worker])
            H['allocated'][sim_ids_to_send] = True  # SH TODO: can do this locally and remove allocated field

            if len(worker_team) > 1:
                Work[worker]['libE_info']['blocking'] = worker_team[1:]  # SH TODO: Maybe do in sim_work?

        else:

            # Allow at most num_active_gens active generator instances
            if gen_count >= alloc_specs['user'].get('num_active_gens', gen_count+1):
                break

            # No gen instances in batch mode if workers still working
            still_working = ~H['returned']
            if alloc_specs['user'].get('batch_mode') and np.any(still_working):
                break

            # Give gen work
            gen_count += 1

            worker_team = assign_workers(avail_workers, 1)  # Returns a list even though one element
            if not worker_team:
                break
            worker = worker_team[0]
            # print('AFTER ASSIGN gen ({}): avail_workers: {}'.format(worker,avail_workers),flush=True)

            if 'in' in gen_specs and len(gen_specs['in']):
                gen_work(Work, worker, gen_specs['in'], range(len(H)), persis_info[worker])
            else:
                gen_work(Work, worker, [], [], persis_info[worker])

    return Work, persis_info
