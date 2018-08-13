from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, os

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from libensemble.message_numbers import EVAL_SIM_TAG 
from libensemble.message_numbers import EVAL_GEN_TAG 

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))

def only_persistent_gens(W, H, sim_specs, gen_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    :See:
        ``/libensemble/tests/regression_tests/test_6-hump_camel_persistent_uniform_sampling.py``
    """
    H = hist.trim_H()
    Work = {}
    gen_count = sum(W['persis_state'] == EVAL_GEN_TAG)
    already_in_Work = np.zeros(len(H),dtype=bool) # To mark points as they are included in Work, but not yet marked as 'given' in H.

    # If i is idle, but in persistent mode, and its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']!=0)]:
        gen_inds = H['gen_worker']==i 
        if np.all(H['returned'][gen_inds]):
            last_ind = np.nonzero(gen_inds)[0][np.argmax(H['given_time'][gen_inds])]
            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': sim_specs['in'] + [name[0] for name in sim_specs['out']],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(last_ind),
                                     'persistent': True
                                }
                       }


    for i in W['worker_id'][np.logical_and(W['active']==0,W['persis_state']==0)]:
        # perform sim evaluations from existing runs (if they exist).
        q_inds_logical = np.logical_and.reduce((~H['given'],~H['paused'],~already_in_Work))

        if np.any(q_inds_logical):
            sim_ids_to_send = np.nonzero(q_inds_logical)[0][0] # oldest point

            Work[i] = {'H_fields': sim_specs['in'],
                       'persis_info': {}, # Our sims don't need information about how points were generatored
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(sim_ids_to_send),
                                },
                      }

            already_in_Work[sim_ids_to_send] = True

        else:
            # Finally, generate points since there is nothing else to do. 
            if gen_count > 0: 
                continue
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

