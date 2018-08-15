#alloc_func

from __future__ import division
from __future__ import absolute_import
import numpy as np

from libensemble.message_numbers import EVAL_GEN_TAG
from libensemble.alloc_funcs.support import avail_worker_ids, sim_work, gen_work

def only_persistent_gens_for_inverse_bayes(W, H, sim_specs, gen_specs, persis_info):
    """
    Starts up to gen_count number of persistent generators.
    These persistent generators produce points (x) in batches and subbatches.
    The points x are given in subbatches to workers to perform a calculation.
    When all subbatches have returned, their output is given back to the
    corresponding persistent generator.

    The first time called there are no persis_w 1st for loop is not done
    """

    Work = {}
    gen_count = sum(W['persis_state'] == EVAL_GEN_TAG)
    already_in_Work = np.zeros(len(H), dtype=bool) # Mark points included in Work, but not 'given' in H.

    # If i is idle, but in persistent mode, and generated work has all returned
    # give output back to i. Otherwise, give nothing to i
    for i in avail_worker_ids(W, persistent=True):
        inds_generated_by_i = (H['gen_worker'] == i) # if > 1 persistant generator, assign the correct work to it
        if np.all(H['returned'][inds_generated_by_i]): # Has sim_f completed everything from this persistent worker?
            # Then give back everything in the last batch
            last_batch_inds = (H['batch'][inds_generated_by_i] == np.max(H['batch'][inds_generated_by_i]))
            inds_to_send_back = np.where(np.logical_and(inds_generated_by_i, last_batch_inds))[0]
            if H['batch'][-1] > 0:
                n = gen_specs['subbatch_size']*gen_specs['num_subbatches']
                k = H['batch'][-1]
                H['weight'][(n*(k-1)):(n*k)] = H['weight'][(n*k):(n*(k+1))]
            gen_work(Work, i, ['like'], persis_info[i],
                     np.atleast_1d(inds_to_send_back), persistent=True)

    for i in avail_worker_ids(W, persistent=False):
        # perform sim evaluations (if any point hasn't been given).
        q_inds_logical = np.logical_and(~H['given'], ~already_in_Work)
        if np.any(q_inds_logical):
            sims_subbatches = H['subbatch'][q_inds_logical]
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][sims_subbatches == np.min(sims_subbatches)]
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send))
            already_in_Work[sim_ids_to_send] = True

        elif gen_count == 0:
            # Finally, generate points since there is nothing else to do.
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], persis_info[i],
                     [], persistent=True)

    return Work, persis_info
