#alloc_func

from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, os

import pdb

from libensemble.message_numbers import EVAL_SIM_TAG 
from libensemble.message_numbers import EVAL_GEN_TAG 




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
    already_in_Work = np.zeros(len(H),dtype=bool) # To mark points as they are included in Work, but not yet marked as 'given' in H.
    
    # If i is idle, but in persistent mode, and generated work has all returned
    # give output back to i. Otherwise, give nothing to i
    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']!=0)]:
        inds_generated_by_i = H['gen_worker']==i #it there is more than 1 persistant generator make sure you assign the correct work to it 
        if np.all(H['returned'][inds_generated_by_i]): # Has sim_f completed everything from this persistent worker?
            # Then give back everything in the last batch
            last_batch_inds = H['batch'][inds_generated_by_i]==np.max(H['batch'][inds_generated_by_i])
            inds_to_send_back = np.where(np.logical_and(inds_generated_by_i,last_batch_inds))[0] 
            if H['batch'][-1] > 0:
                n = gen_specs['subbatch_size']*gen_specs['num_subbatches']
                k = H['batch'][-1]
                H['weight'][(n*(k-1)):(n*k)] = H['weight'][(n*k):(n*(k+1))]    
            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': ['like'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(inds_to_send_back), #atleast_1d -> Convert inputs to arrays with at least one dimension.
                                     'persistent': True
                                }
                       }

    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']==0)]:
        # perform sim evaluations (if any point hasn't been given).
        q_inds_logical = np.logical_and(~H['given'],~already_in_Work) 
        if np.any(q_inds_logical):
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][H['subbatch'][q_inds_logical]==np.min(H['subbatch'][q_inds_logical])]
            Work[i] = {'H_fields': sim_specs['in'], #things to evaluate
                       'persis_info': {}, # Our sims don't need information about how points were generated
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(sim_ids_to_send), #tells me what x's the returned values go with
                                },
                      }

            already_in_Work[sim_ids_to_send] = True

        else:
            # Finally, generate points since there is nothing else to do. 
            if gen_count > 0: 
                continue # continue with the next loop of the iteration
            gen_count += 1
            # There are no points available, so we call our gen_func
            Work[i] = {'persis_info':persis_info[i],
                       'H_fields': gen_specs['in'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': [],
                                     'persistent': True
                                }

                       }
     

    return Work, persis_info

