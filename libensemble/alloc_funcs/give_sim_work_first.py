import numpy as np

# SH TODO:  Consider importing a class and using as object functions
from libensemble.tools.alloc_support import AllocSupport


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

    Can give points in highest priority, if ``'priority'`` is a field in ``H``.
    If gen_specs['user']['give_all_with_same_priority'] is set to True, then
    all points with the same priority value are given as a batch to the sim.

    Workers performing sims will be assigned resources given by the resource_sets
    field of H if it exists, else defaulting to one. Workers performing gens are
    assigned resource_sets given by persis_info['gen_resources'] or zero.

    This is the default allocation function if one is not defined.

    .. seealso::
        `test_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_uniform_sampling.py>`_ # noqa
    """

    support = AllocSupport()  # Access alloc support functions

    Work = {}
    gen_count = support.count_gens(W)

    task_avail = ~H['given'] & ~H['cancel_requested']
    avail_workers = support.avail_worker_ids(W)

    while avail_workers:

        if np.any(task_avail):
            # Pick all high priority, oldest high priority, or just oldest point
            if 'priority' in H.dtype.fields:
                priorities = H['priority'][task_avail]
                if gen_specs['user'].get('give_all_with_same_priority'):
                    q_inds = (priorities == np.max(priorities))
                else:
                    q_inds = np.argmax(priorities)
            else:
                q_inds = 0

            # Get sim ids (indices) and check resources needed
            sim_ids_to_send = np.nonzero(task_avail)[0][q_inds]  # oldest point(s)

            num_rsets_req = (np.max(H[sim_ids_to_send]['resource_sets'])
                             if 'resource_sets' in H.dtype.names else 1)

            # If more than one group (node) required, allocates whole nodes - also removes from avail_workers
            print('\nrset_team being called for sim. Requesting {} rsets'.format(num_rsets_req))

            # SH TODO: Gives worker ID so rsets can be set - but doesn't remove from avail_workers
            rset_team = support.assign_resources(num_rsets_req)

            # print('AFTER ASSIGN sim ({}): avail_workers: {}'.format(worker_team,avail_workers),flush=True)

            # None means insufficient available resources for this work unit
            if rset_team is None:
                break

            # Assign points to worker and remove from task_avail list.
            # SH TODO: With alloc_support - can store workers in alloc_suport and easily combine this...
            #          - have to also combine with breaking if rset_team is None (shld rset_team be nrsets??)
            #          - and remember we want to combine with packing rset team in sim_work - as not going to assign...
            worker = avail_workers.pop(0)  # Give to first worker in list
            print('resource team for SIM {} assigned to worker {}'.format(rset_team, worker), flush=True)

            support.sim_work(Work, worker, sim_specs['in'], sim_ids_to_send, persis_info.get(worker), rset_team=rset_team)
            task_avail[sim_ids_to_send] = False
        else:

            # Allow at most num_active_gens active generator instances
            if gen_count >= alloc_specs['user'].get('num_active_gens', gen_count+1):
                break

            # Do not start gen instances in batch mode if workers still working
            if alloc_specs['user'].get('batch_mode') and not support.all_returned(H):
                break

            # Give gen work
            gen_count += 1

            gen_resources = persis_info.get('gen_resources', 0)
            rset_team = support.assign_resources(gen_resources)

            # None means insufficient available resources for this work unit
            if rset_team is None:
                break

            worker = avail_workers.pop(0)  # Give to first worker in list
            print('resource team for GEN {} assigned to worker {}'.format(rset_team, worker), flush=True)

            # print('AFTER ASSIGN gen ({}): avail_workers: {}'.format(worker,avail_workers),flush=True)

            gen_in = gen_specs.get('in', [])
            return_rows = range(len(H)) if gen_in else []
            support.gen_work(Work, worker, gen_in, return_rows, persis_info.get(worker), rset_team=rset_team)

    return Work, persis_info
