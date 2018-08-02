from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, os

from libensemble.message_numbers import EVAL_SIM_TAG 
from libensemble.message_numbers import EVAL_GEN_TAG 

def give_sim_work_first(W, H, sim_specs, gen_specs, persis_info):
    """ 
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most gen_specs['num_inst'])
    generator instances.
    
    note: everything put into the Work dictionary will be given, so be
    careful not to put more gen or sim items into Work than necessary.

    Parameters
    -----------
    H: numpy structured array

    H_ind: integer 

    sim_specs: dictionary

    gen_specs: dictionary

    term_test: lambda function

    persis_info: dictionary

    Returns
    -----------
    Work: dictionary
        Each integer key corresponds to a worker that will be given the
        corresponding dictionary values
    
    persis_info: dictionary
        Updated generation informaiton 
    """

    Work = {}
    gen_count = 0

    for i in W['nonpersis_w']['waiting']:

        # Find indices of H where that are not yet allocated
        if persis_info['next_to_give'] < len(H):
            # Give sim work if possible
            Work[i] = {'H_fields': sim_specs['in'],
                       'persis_info': {}, # Our sims don't need information about how points were generatored
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': [persis_info['next_to_give']],
                                },
                      }
            persis_info['next_to_give'] += 1

        else:
            # Since there is no sim work to give, give gen work...

            # ...unless there are already more than num_active_gen instances
            if 'num_active_gens' in gen_specs and len(W['nonpersis_w'][EVAL_GEN_TAG]) + gen_count >= gen_specs['num_active_gens']:
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

