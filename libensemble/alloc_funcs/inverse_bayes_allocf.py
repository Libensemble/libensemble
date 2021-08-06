import numpy as np

from libensemble.tools.alloc_support import (avail_worker_ids, sim_work, gen_work,
                                             count_persis_gens, all_returned)


def only_persistent_gens_for_inverse_bayes(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    Starts up to gen_count number of persistent generators.
    These persistent generators produce points (x) in batches and subbatches.
    The points x are given in subbatches to workers to perform a calculation.
    When all subbatches have returned, their output is given back to the
    corresponding persistent generator.

    The first time called there are no persis_w 1st for loop is not done
    """

    Work = {}
    gen_count = count_persis_gens(W)

    # If i is idle, but in persistent mode, and generated work has all returned
    # give output back to i. Otherwise, give nothing to i
    for i in avail_worker_ids(W, persistent=True):

        # if > 1 persistant generator, assign the correct work to it
        inds_generated_by_i = (H['gen_worker'] == i)
        if all_returned(H, inds_generated_by_i):

            # Has sim_f completed everything from this persistent worker?
            # Then give back everything in the last batch
            batch_ids = H['batch'][inds_generated_by_i]
            last_batch_inds = (batch_ids == np.max(batch_ids))
            inds_to_send_back = np.where(np.logical_and(inds_generated_by_i,
                                                        last_batch_inds))[0]
            if H['batch'][-1] > 0:
                n = gen_specs['user']['subbatch_size']*gen_specs['user']['num_subbatches']
                k = H['batch'][-1]
                H['weight'][(n*(k-1)):(n*k)] = H['weight'][(n*k):(n*(k+1))]

            gen_work(Work, i, ['like'], inds_to_send_back,
                     persis_info.get(i), persistent=True)

    task_avail = ~H['given'] & ~H['cancel_requested']
    for i in avail_worker_ids(W, persistent=False):
        if np.any(task_avail):

            # perform sim evaluations (if any point hasn't been given).
            sim_subbatches = H['subbatch'][task_avail]
            sim_inds = (sim_subbatches == np.min(sim_subbatches))
            sim_ids_to_send = np.nonzero(task_avail)[0][sim_inds]
            sim_work(Work, i, sim_specs['in'], sim_ids_to_send, [])
            task_avail[sim_ids_to_send] = False

        elif gen_count == 0:

            # Finally, generate points since there is nothing else to do.
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], [], persis_info.get(i),
                     persistent=True)

    return Work, persis_info
