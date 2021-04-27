import numpy as np

from libensemble.tools.alloc_support import avail_worker_ids, sim_work, gen_work, count_persis_gens, all_returned


def persistent_aposmm_alloc(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start a persistent APOSMM generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    This function assumes that one persistent APOSMM will be started and never
    stopped (until some exit_criterion is satisfied).

    .. seealso::
        `test_persistent_aposmm_with_grad.py <https://github.com/Libensemble/libensemble/blob/develop/libensemble/tests/regression_tests/test_persistent_aposmm_with_grad.py>`_ # noqa
    """

    Work = {}
    gen_count = count_persis_gens(W)

    if persis_info.get('first_call', True):
        assert np.all(H['given']), "Initial points in H have never been given."
        assert np.all(H['given_back']), "Initial points in H have never been given_back."
        assert all_returned(H), "Initial points in H have never been returned."
        persis_info['fields_to_give_back'] = ['f'] + [n[0] for n in gen_specs['out']]

        if 'grad' in [n[0] for n in sim_specs['out']]:
            persis_info['fields_to_give_back'] += ['grad']

        if 'fvec' in [n[0] for n in sim_specs['out']]:
            persis_info['fields_to_give_back'] += ['fvec']

        persis_info['samples_in_H0'] = sum(H['local_pt'] == 0)
        persis_info['next_to_give'] = len(H)  #
        persis_info['first_call'] = False
    elif gen_count == 0:
        # The one persistent gen is done. Exiting
        return Work, persis_info, 1

    # If any persistent worker's calculated values have returned, give them back.
    for i in avail_worker_ids(W, persistent=True):
        if (persis_info.get('sample_done') or
           sum(H['returned']) >= gen_specs['user']['initial_sample_size'] + persis_info['samples_in_H0']):
            # Don't return if the initial sample is not complete
            persis_info['sample_done'] = True

            returned_but_not_given = np.logical_and(H['returned'], ~H['given_back'])
            if np.any(returned_but_not_given):
                inds_to_give = np.where(returned_but_not_given)[0]

                gen_work(Work, i, persis_info['fields_to_give_back'],
                         np.atleast_1d(inds_to_give), persis_info.get(i), persistent=True)

                H['given_back'][inds_to_give] = True

    for i in avail_worker_ids(W, persistent=False):
        # Skip any cancelled points
        while persis_info['next_to_give'] < len(H) and H[persis_info['next_to_give']]['cancel_requested']:
            persis_info['next_to_give'] += 1

        if persis_info['next_to_give'] < len(H):
            # perform sim evaluations (if they exist in History).
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(persis_info['next_to_give']), persis_info.get(i))
            persis_info['next_to_give'] += 1

        elif persis_info.get('gen_started') is None:
            # Finally, call a persistent generator as there is nothing else to do.
            persis_info['gen_started'] = True
            persis_info.get(i)['nworkers'] = len(W)

            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info.get(i),
                     persistent=True)

    return Work, persis_info
