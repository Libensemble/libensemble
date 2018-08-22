from __future__ import division
from __future__ import absolute_import
import numpy as np

from libensemble.alloc_funcs.support import \
     avail_worker_ids, sim_work, gen_work, count_persis_gens


def only_persistent_gens(W, H, sim_specs, gen_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    :See:
        ``/libensemble/tests/regression_tests/test_6-hump_camel_persistent_uniform_sampling.py``
    """

    Work = {}
    gen_count = count_persis_gens(W)

    # If i is idle, but in persistent mode, and its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in avail_worker_ids(W, persistent=True):
        gen_inds = (H['gen_worker'] == i)
        if np.all(H['returned'][gen_inds]):
            last_time_pos = np.argmax(H['given_time'][gen_inds])
            last_ind = np.nonzero(gen_inds)[0][last_time_pos]
            gen_work(Work, i,
                     sim_specs['in'] + [n[0] for n in sim_specs['out']],
                     persis_info[i], np.atleast_1d(last_ind), persistent=True)

    task_avail = ~H['given']
    for i in avail_worker_ids(W, persistent=False):
        if np.any(task_avail):

            # perform sim evaluations from existing runs (if they exist).
            sim_ids_to_send = np.nonzero(task_avail)[0][0] # oldest point
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send))
            task_avail[sim_ids_to_send] = False

        elif gen_count == 0:
            # Finally, generate points since there is nothing else to do.
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], persis_info[i],
                     [], persistent=True)

    return Work, persis_info
