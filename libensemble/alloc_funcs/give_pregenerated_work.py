from __future__ import division
from __future__ import absolute_import

from libensemble.alloc_funcs.support import \
     avail_worker_ids, sim_work, gen_work, count_gens


def give_pregenerated_sim_work(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in alloc_spec['x'] to
    idle workers. It is an example use case where no gen_func is used. 

    :See:
        ``/libensemble/tests/regression_tests/test_fast_alloc.py``
    """

    Work = {}
    if not persis_info:
        persis_info['next_to_give'] = 0

    for i in avail_worker_ids(W):
        if persis_info['next_to_give'] < len(H):

            # Give sim work 
            sim_work(Work, i, sim_specs['in'], [persis_info['next_to_give']], [])
            persis_info['next_to_give'] += 1

        else: 
            print('No more work to give inside give_pregenerated_sim_work.') 


    return Work, persis_info
