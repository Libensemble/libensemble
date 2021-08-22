import numpy as np
from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.tools.alloc_support import AllocSupport, InsufficientFreeResources


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

    # Initialize alloc_specs['user'] as user.
    user = alloc_specs.get('user', {})
    sched_opts = user.get('scheduler_opts', {})

    # SH TODO: See below -- gen_specs['user']['initial_sample_size']
    #init_sample_size = user.get('init_sample_size', 0)   # LINE FROM start_only_persistent

    # SH TODO: Does not do priority (uses next_to_give) - do we want it to do priority??..
    #batch_give = gen_specs['user'].get('give_all_with_same_priority', False)  # SH TODO: Should this be alloc_specs not gen_specs?

    support = AllocSupport(W, H, persis_info, sched_opts)
    gen_count = support.count_persis_gens()
    Work = {}

    if persis_info.get('first_call', True):
        # SH TODO: Checking initial points completed... What if were cancelled?
        assert np.all(H['given']), "Initial points in H have never been given."
        assert np.all(H['given_back']), "Initial points in H have never been given_back."
        assert support.all_returned(H), "Initial points in H have never been returned."

        # SH TODO: gen_specs['persis_in'] - and why is done here and in persis_info - does this vary?
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
    for wid in support.avail_worker_ids(persistent=EVAL_GEN_TAG):
        if (persis_info.get('sample_done') or
           sum(H['returned']) >= gen_specs['user']['initial_sample_size'] + persis_info['samples_in_H0']):
            # Don't return if the initial sample is not complete
            persis_info['sample_done'] = True

            #returned_but_not_given = np.logical_and(H['returned'], ~H['given_back'])
            points_evaluated = support.get_evaluated_points(gen=wid)

            if np.any(points_evaluated):
                point_ids = np.where(points_evaluated)[0]
                #rset_team = [] if manage_resources else None
                support.gen_work(Work, wid, persis_info['fields_to_give_back'],
                                 point_ids, persis_info.get(wid), persistent=True)
                H['given_back'][point_ids] = True   # SH TODO: Move to manager (libE field)
                points_evaluated[point_ids] = False  # SH TODO: May not need if each iteration is mutually exclusive points!

    # SH TODO: Now the give_sim_work_first bit - but with 'next_to_give'
    # SH TODO: 1. Give the same structure as start_only_persistent zrw=False loop and zrw=true loop.
    #          2. Decide whether we change to allow >1 gen (if not - dont need a 2nd loop)
    #          3. Okay to add active_recv=active_recv_gen option??? (and should it be default).
    for wid in support.avail_worker_ids(persistent=False):
        # Skip any cancelled points
        while persis_info['next_to_give'] < len(H) and H[persis_info['next_to_give']]['cancel_requested']:
            persis_info['next_to_give'] += 1

        if persis_info['next_to_give'] < len(H):
            # perform sim evaluations (if they exist in History).
            try:
                support.sim_work(Work, wid, sim_specs['in'], persis_info['next_to_give'], persis_info.get(wid))
            except InsufficientFreeResources:
                break
            persis_info['next_to_give'] += 1

        elif persis_info.get('gen_started') is None:
            # Finally, call a persistent generator as there is nothing else to do.
            persis_info.get(wid)['nworkers'] = len(W)
            try:
                support.gen_work(Work, wid, gen_specs['in'], range(len(H)), persis_info.get(wid),
                                 persistent=True)  #, active_recv=active_recv_gen)
            except InsufficientFreeResources:
                break
            persis_info['gen_started'] = True  # Must set after - incase break on resources

    return Work, persis_info
