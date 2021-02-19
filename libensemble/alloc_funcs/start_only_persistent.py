import numpy as np
from libensemble.tools.alloc_support import (avail_worker_ids,
                                             sim_work, gen_work,
                                             count_persis_gens,
                                             get_avail_workers_by_group,
                                             assign_workers)


# SH TODO: Either replace only_persistent_gens or add a different alloc func (or file?)
#          Check/update docstring
def only_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
        `test_persistent_uniform_sampling_async.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling_async.py>`_ # noqa
    """

    Work = {}
    gen_count = count_persis_gens(W)

    if persis_info.get('gen_started') and gen_count == 0:
        # The one persistent worker is done. Exiting
        return Work, persis_info, 1

    for i in avail_worker_ids(W, persistent=True):
        if gen_specs['user'].get('async', False):
            # If i is in persistent mode, asynchronous behavior is desired, and
            # *any* of its calculated values have returned, give them back to i.
            # Otherwise, give nothing to i
            returned_but_not_given = np.logical_and.reduce((H['returned'], ~H['given_back'], H['gen_worker'] == i))
            if np.any(returned_but_not_given):
                inds_to_give = np.where(returned_but_not_given)[0]
                gen_work(Work, i,
                         sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')],
                         np.atleast_1d(inds_to_give), persis_info[i], persistent=True)

                H['given_back'][inds_to_give] = True

        else:
            # If i is in persistent mode, batch behavior is desired, and
            # *all* of its calculated values have returned, give them back to i.
            # Otherwise, give nothing to i
            gen_inds = (H['gen_worker'] == i)
            if np.all(H['returned'][gen_inds]):
                last_time_gen_gave_batch = np.max(H['gen_time'][gen_inds])
                inds_to_give = H['sim_id'][gen_inds][H['gen_time'][gen_inds] == last_time_gen_gave_batch]
                gen_work(Work, i,
                         sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')],
                         np.atleast_1d(inds_to_give), persis_info[i], persistent=True)

                H['given_back'][inds_to_give] = True

    task_avail = ~H['given']  # SH TODO: Unchanged - but what if allocated space in array that is not genned?

    # SH TODO: Now the give_sim_work_first bit - should merge to avoid duplicating functionality
    avail_workers = get_avail_workers_by_group(W, persistent=False, zero_resource_workers=False)
    # print('avail_workers for sim',avail_workers)

    while any(avail_workers.values()):

        if not np.any(task_avail):
            break

        # Perform sim evaluations (if they exist in History).
        sim_ids_to_send = np.nonzero(task_avail)[0][0]  # oldest point

        nworkers_req = (np.max(H[sim_ids_to_send]['resource_sets'])
                        if 'resource_sets' in H.dtype.names else 1)

        # If more than one group (node) required, allocates whole nodes - also removes from avail_workers
        worker_team = assign_workers(avail_workers, nworkers_req)
        # print('AFTER ASSIGN sim ({}): avail_workers: {}'.format(worker_team,avail_workers), flush=True)

        if not worker_team:
            break  # No slot found - insufficient available resources for this work unit

        worker = worker_team[0]
        sim_work(Work, worker, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info[worker])

        task_avail[sim_ids_to_send] = False
        if len(worker_team) > 1:
            Work[worker]['libE_info']['blocking'] = worker_team[1:]  # SH TODO: Maybe do in sim_work?

    # A separate loop/section as now need zero_resource_workers for gen.
    if not np.any(task_avail):
        avail_workers = get_avail_workers_by_group(W, persistent=False, zero_resource_workers=True)
        # print('avail_workers for gen',avail_workers)

        while any(avail_workers.values()):
            # SH TODO: So we don't really need a loop here for this, but general case would allow multiple gens
            if gen_count == 0:
                # Finally, call a persistent generator as there is nothing else to do.
                gen_count += 1

                worker_team = assign_workers(avail_workers, 1)  # Returns a list even though one element
                if not worker_team:
                    break
                worker = worker_team[0]
                # print('AFTER ASSIGN gen ({}): avail_workers: {}'.format(worker,avail_workers), flush=True)
                gen_work(Work, worker, gen_specs['in'], range(len(H)), persis_info[worker],
                         persistent=True)
                persis_info['gen_started'] = True

    return Work, persis_info, 0
