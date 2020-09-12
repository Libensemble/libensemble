import numpy as np

from libensemble.tools.alloc_support import avail_worker_ids, sim_work, gen_work, count_persis_gens


def only_persistent_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  By default (batch_mode is
    True), when all points requested by the persistent generator have been
    returned from the simulation evaluation, then this information is given
    back to the persistent generator. If batch_mode is False, then any returned
    points are given back to the generator.

    Batch mode is determined by ``alloc_specs['user']['batch_mode']`

    .. seealso::
        `test_persistent_uniform_sampling.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_uniform_sampling.py>`_ # noqa
    """

    Work = {}
    gen_count = count_persis_gens(W)

    # Initialize alloc_specs['user'] if not set.
    alloc_specs['user'] = alloc_specs.get('user', {})

    # In batch_mode (default), gen called only when all evaluations have returned.
    batch_mode = alloc_specs['user'].get('batch_mode', True)  # Defaults to true
    batch_to_sim_id = alloc_specs['user'].get('batch_to_sim_id', -1)  # Always do batch up to this sim_id

    if persis_info.get('gen_started') and gen_count == 0:
        # The one persistent worker is done. Exiting
        return Work, persis_info, 1

    # If i is in persistent mode, and all of its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in avail_worker_ids(W, persistent=True):
        gen_inds = (H['gen_worker'] == i)
        inds_since_last_gen = H['sim_id'][gen_inds][H['returned'] & ~H['given_back']]

        give_back_to_gen = False
        if inds_since_last_gen.size > 0:
            if max(inds_since_last_gen) < batch_to_sim_id or batch_mode:
                if np.all(H['returned'][gen_inds]):
                    give_back_to_gen = True
            else:
                give_back_to_gen = True

        if give_back_to_gen:
            gen_work(Work, i,
                     sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')],
                     np.atleast_1d(inds_since_last_gen), persis_info[i], persistent=True)
            H['given_back'][inds_since_last_gen] = True

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
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info[i],
                     persistent=True)
            persis_info['gen_started'] = True

    return Work, persis_info, 0
