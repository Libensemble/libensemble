import numpy as np

from libensemble.alloc_funcs.support import avail_worker_ids, sim_work, gen_work, count_persis_gens


def finite_diff_alloc(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start 1 persistent generator.  If all points requested by
    the persistent generator for a given (x_ind,f_ind) pair have been returned from the
    simulation evaluation, then this information is given back to the
    persistent generator (where x_ind is in range(n) and f_ind is in range(p))

    .. seealso::
        `test_persist_fd_param_finder.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persist_fd_param_finder.py>`_
    """

    Work = {}
    gen_count = count_persis_gens(W)

    if len(H) and gen_count == 0:
        # The one persistent worker is done. Exiting
        return Work, persis_info, 1

    # If i is in persistent mode, and all of its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in avail_worker_ids(W, persistent=True):
        gen_inds = (H['gen_worker'] == i)
        if np.all(H['returned'][gen_inds]):
            last_time_gen_gave_batch = np.max(H['gen_time'][gen_inds])
            inds_of_last_batch_from_gen = H['sim_id'][gen_inds][H['gen_time'][gen_inds] == last_time_gen_gave_batch]
            gen_work(Work, i,
                     sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')],
                     np.atleast_1d(inds_of_last_batch_from_gen), persis_info[i], persistent=True)

            H['given_back'][inds_of_last_batch_from_gen] = True

    task_avail = ~H['given']
    for i in avail_worker_ids(W, persistent=False):
        if np.any(task_avail):
            # perform sim evaluations (if they exist in History).
            sim_ids_to_send = np.nonzero(task_avail)[0][0]  # oldest point
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info[i])
            task_avail[sim_ids_to_send] = False

        elif gen_count == 0:
            # Finally, call a persistent generator as there is nothing else to do.
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], [], persis_info[i],
                     persistent=True)

    return Work, persis_info, 0
