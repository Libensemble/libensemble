from __future__ import division
from __future__ import absolute_import

import numpy as np

from libensemble.alloc_funcs.support import \
     avail_worker_ids, sim_work, gen_work, count_gens

def give_sim_work_first(W, H, sim_specs, gen_specs, persis_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If all entries in `H` have been given a be evaluated, a worker
    is told to call the generator function, provided this wouldn't result in
    more than ``gen_specs['num_active_gen']`` active generators. Also allows
    for a 'batch_mode'.

    :See:
        ``/libensemble/tests/regression_tests/test_6-hump_camel_aposmm_LD_MMA.py``
    """

    Work = {}
    gen_count = count_gens(W)

    for i in avail_worker_ids(W):

        # Find indices of H that are not yet allocated
        if persis_info['next_to_give'] < len(H):
            # Give sim work if possible
            sim_work(Work, i, sim_specs['in'], [persis_info['next_to_give']])
            persis_info['next_to_give'] += 1

        elif gen_count < gen_specs.get('num_active_gens', gen_count+1):

            # Don't give gen instances in batch mode if points are unfinished
            last_size = persis_info.get('last_size')
            if (gen_specs.get('batch_mode')
                    and len(H)
                    and not all(np.logical_or(H['returned'][last_size:],
                                              H['paused'][last_size:]))):
                break
            else:
                persis_info['last_size'] = len(H)

            # Give gen work
            persis_info['total_gen_calls'] += 1
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], persis_info[i], range(len(H)))

    return Work, persis_info
