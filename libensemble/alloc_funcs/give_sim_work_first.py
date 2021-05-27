import numpy as np
# SH TODO:  Consider importing a class and using as object functions
from libensemble.tools.alloc_support import (sim_work, gen_work, count_gens,
                                             avail_worker_ids, assign_resources)


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
    avail_workers = avail_worker_ids(W)

    while avail_workers:

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

            num_rsets_req = (np.max(H[sim_ids_to_send]['resource_sets'])
                            if 'resource_sets' in H.dtype.names else 1)

            # If more than one group (node) required, allocates whole nodes - also removes from avail_workers
            print('\nrset_team being called for sim. Requesting {} rsets'.format(num_rsets_req))

            rset_team = assign_resources(num_rsets_req, avail_workers[0])  #So it can tell resources.rsets what worker

            # print('AFTER ASSIGN sim ({}): avail_workers: {}'.format(worker_team,avail_workers),flush=True)

            # None means insufficient available resources for this work unit
            if rset_team is None:
                break

            worker = avail_workers.pop(0)  # Give to first worker in list
            print('resource team for SIM {} assigned to worker {}'.format(rset_team,worker), flush=True)

            sim_work(Work, worker, sim_specs['in'], sim_ids_to_send, persis_info[worker])
            H['allocated'][sim_ids_to_send] = True  # SH TODO: can do this locally and remove allocated field
            Work[worker]['libE_info']['rset_team'] = rset_team
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
            gen_resources = persis_info.get('gen_resources', 0)
            rset_team = assign_resources(gen_resources, avail_workers[0])

            # None means insufficient available resources for this work unit
            if rset_team is None:
                break

            worker = avail_workers.pop(0)  # Give to first worker in list
            print('resource team for GEN {} assigned to worker {}'.format(rset_team,worker), flush=True)

            # print('AFTER ASSIGN gen ({}): avail_workers: {}'.format(worker,avail_workers),flush=True)

            if 'in' in gen_specs and len(gen_specs['in']):
                gen_work(Work, worker, gen_specs['in'], range(len(H)), persis_info[worker])
            else:
                gen_work(Work, worker, [], [], persis_info[worker])

            Work[worker]['libE_info']['rset_team'] = rset_team  #could be zero - but sending tells it to use this and not index_list

    return Work, persis_info
