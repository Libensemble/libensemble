import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.alloc_funcs.support import avail_worker_ids, sim_work, gen_work, count_persis_gens

from libensemble.gen_funcs.aposmm import initialize_APOSMM, decide_where_to_start_localopt, update_history_dist


def start_persistent_local_opt_gens(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will:

    - Start up a persistent generator that is a local opt run at the first point
      identified by APOSMM's decide_where_to_start_localopt.
    - It will only do this if at least one worker will be left to perform
      simulation evaluations.
    - If multiple starting points are available, the one with smallest function
      value is chosen.
    - If no candidate starting points exist, points from existing runs will be
      evaluated (oldest first).
    - If no points are left, call the generation function.

    :See:
        ``/libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling_with_persistent_localopt_gens.py``
    """

    Work = {}
    gen_count = count_persis_gens(W)
    task_avail = ~H['given']

    # If a persistent localopt run has just finished, use run_order to update H
    # and then remove other information from persis_info
    for i in persis_info.keys():
        if 'done' in persis_info[i]:
            H['num_active_runs'][persis_info[i]['run_order']] -= 1
        if 'x_opt' in persis_info[i]:
            opt_ind = np.all(H['x'] == persis_info[i]['x_opt'], axis=1)
            assert sum(opt_ind) == 1, "There must be just one optimum"
            H['local_min'][opt_ind] = True
            persis_info[i] = {'rand_stream': persis_info[i]['rand_stream']}

    # If i is idle, but in persistent mode, and its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in avail_worker_ids(W, persistent=True):
        gen_inds = (H['gen_worker'] == i)
        if np.all(H['returned'][gen_inds]):
            last_time_pos = np.argmax(H['given_time'][gen_inds])
            last_ind = np.nonzero(gen_inds)[0][last_time_pos]
            gen_work(Work, i,
                     sim_specs['in'] + [n[0] for n in sim_specs['out']],
                     np.atleast_1d(last_ind), persis_info[i], persistent=True)
            persis_info[i]['run_order'].append(last_ind)

    for i in avail_worker_ids(W, persistent=False):
        # Find candidates to start local opt runs if a sample has been evaluated
        if np.any(np.logical_and(~H['local_pt'], H['returned'])):
            n, _, _, _, r_k, mu, nu = initialize_APOSMM(H, gen_specs)
            update_history_dist(H, n, gen_specs, c_flag=False)
            starting_inds = decide_where_to_start_localopt(H, r_k, mu, nu)
        else:
            starting_inds = []

        # Start persistent generator for local opt run unless it would use all workers
        if starting_inds and gen_count + 1 < len(W):
            # Start at the best possible starting point
            ind = starting_inds[np.argmin(H['f'][starting_inds])]
            gen_work(Work, i,
                     sim_specs['in'] + [n[0] for n in sim_specs['out']],
                     np.atleast_1d(ind), persis_info[i], persistent=True)

            H['started_run'][ind] = 1
            H['num_active_runs'][ind] += 1

            persis_info[i]['run_order'] = [ind]
            gen_count += 1

        elif np.any(task_avail):

            # Perform sim evaluations from existing runs
            q_inds_logical = np.logical_and(task_avail, H['local_pt'])
            if not np.any(q_inds_logical):
                q_inds_logical = task_avail
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][0]  # oldest point
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), [])
            task_avail[sim_ids_to_send] = False

        elif (gen_count == 0
              and not np.any(np.logical_and(W['active'] == EVAL_GEN_TAG,
                                            W['persis_state'] == 0))):

            # Finally, generate points since there is nothing else to do
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], [], persis_info[i])

    return Work, persis_info
