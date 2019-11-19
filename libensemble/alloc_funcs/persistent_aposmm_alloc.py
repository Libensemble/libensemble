import numpy as np

from libensemble.alloc_funcs.support import avail_worker_ids, sim_work, gen_work


def persistent_aposmm_alloc(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start a persistent APOSMM generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    This function assumes that one persistent APOSMM will be started and never
    stopped (until some exit_criterion is satisfied).

    .. seealso::
        `test_6-hump_camel_persistent_aposmm.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_6-hump_camel_persistent_aposmm.py>`_
    """

    Work = {}
    if 'next_to_give' not in persis_info:
        persis_info['next_to_give'] = 0

    # If any persistent worker's calculated values have returned, give them back.
    for i in avail_worker_ids(W, persistent=True):
        if persis_info.get('sample_done') or sum(H['returned']) >= gen_specs['user']['initial_sample_size']:
            # Don't return if the initial sample is not complete
            persis_info['sample_done'] = True

            returned_but_not_given = np.logical_and(H['returned'], ~H['given_back'])
            if np.any(returned_but_not_given):
                inds_to_give = np.where(returned_but_not_given)[0]

                gen_work(Work, i, [n[0] for n in sim_specs['out']] + [n[0] for n in gen_specs['out']],
                         np.atleast_1d(inds_to_give), persis_info[i], persistent=True)

                H['given_back'][inds_to_give] = True

    for i in avail_worker_ids(W, persistent=False):
        if persis_info['next_to_give'] < len(H):
            # perform sim evaluations (if they exist in History).
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(persis_info['next_to_give']), persis_info[i])
            persis_info['next_to_give'] += 1

        elif persis_info.get('gen_started') is None:
            # Finally, call a persistent generator as there is nothing else to do.
            persis_info['gen_started'] = True

            gen_work(Work, i, gen_specs['in'], [], persis_info[i],
                     persistent=True)

    return Work, persis_info
