from __future__ import division
from __future__ import absolute_import

from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG


def give_sim_work_first(W, H, sim_specs, gen_specs, persis_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If all entries in `H` have been given a be evaluated, a worker
    is told to call the generator function, provided this wouldn't result in
    more than ``gen_specs['num_active_gen']`` active generators.

    :See:
        ``/libensemble/tests/regression_tests/test_fast_alloc.py``
    """

    Work = {}
    gen_count = sum(W['active'] == EVAL_GEN_TAG)

    for i in W['worker_id'][W['active'] == 0]:

        # Find indices of H that are not yet allocated
        if persis_info['next_to_give'] < len(H):
            # Give sim work if possible
            Work[i] = {'H_fields': sim_specs['in'],
                       'persis_info': {}, # Our sims don't need information about how points were generatored
                       'tag':EVAL_SIM_TAG,
                       'libE_info': {'H_rows': [persis_info['next_to_give']]},
                      }
            persis_info['next_to_give'] += 1

        else:
            # Since there is no sim work to give, give gen work...

            # ...unless there are already more than num_active_gen instances
            if 'num_active_gens' in gen_specs and gen_count >= gen_specs['num_active_gens']:
                break

            # Give gen work
            persis_info['total_gen_calls'] += 1
            gen_count += 1

            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': gen_specs['in'],
                       'tag': EVAL_GEN_TAG,
                       'libE_info': {'H_rows': []}
                      }

    return Work, persis_info
