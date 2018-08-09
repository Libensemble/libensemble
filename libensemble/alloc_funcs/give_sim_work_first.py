from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, os

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from libensemble.message_numbers import EVAL_SIM_TAG 
from libensemble.message_numbers import EVAL_GEN_TAG 

def give_sim_work_first(W, H, sim_specs, gen_specs, persis_info):
    """ 
    Decide what should be given to workers. This allocation function gives any
    available simulation work first, and only when all simulations are
    completed or running does it start (at most ``gen_specs['num_inst']``)
    generator instances. 
    
    Allows for a ``'batch_mode'`` where no generation
    work is given out unless all entries in ``H`` are either returned or
    paused. 
    
    Allows for ``blocking`` of workers that are not active, for example, so
    their resources can be used for a different simulation evaluation.
    
    Can give points in highest priority, if ``'priority'`` is a field in ``H``. 
    
    This is the default allocation function if one is not defined.

    :See: 
        ``/libensemble/tests/regression_tests/test_6-hump_camel_uniform_sampling.py``
    """

    Work = {}
    gen_count = sum(W['active'] == EVAL_GEN_TAG)

    for i in W['worker_id'][W['active']==0]:

        # Only consider giving to worker i if it's resources are not blocked by some other calculation
        blocked_set = set(W['worker_id'][W['blocked']]).union(*[j['libE_info']['blocking'] for j in Work.values() if 'blocking' in j['libE_info']])
        if i in blocked_set:
            continue

        # Find indices of H that are not given nor paused
        jj = list(H['allocated'])
        if not all(jj):
            # Give sim work if possible

            if 'priority' in H.dtype.fields:
                if 'give_all_with_same_priority' in gen_specs and gen_specs['give_all_with_same_priority']:
                    # Give all points with highest priority
                    q_inds = H['priority'][~H['allocated']] == np.max(H['priority'][~H['allocated']])
                    sim_ids_to_send = np.nonzero(~H['allocated'])[0][q_inds]
                else:
                    # Give first point with highest priority
                    sim_ids_to_send = np.nonzero(~H['allocated'])[0][np.argmax(H['priority'][~H['allocated']])]
            else:
                # Give oldest point
                sim_ids_to_send = np.nonzero(~H['allocated'])[0][0]

            sim_ids_to_send = np.atleast_1d(sim_ids_to_send)

            # Only give work if enough idle workers
            if 'num_nodes' in H.dtype.names and np.any(H[sim_ids_to_send]['num_nodes'] > 1):
                if np.any(H[sim_ids_to_send]['num_nodes'] > sum(W['active']==0) - len(Work) - len(blocked_set)):
                    # Worker i doesn't get any work. Just waiting for other resources to open up
                    continue
                block_others = True
            else:
                block_others = False

            Work[i] = {'H_fields': sim_specs['in'],
                       'persis_info': {}, # Our sims don't need information about how points were generatored
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': sim_ids_to_send,
                                },
                      }
            H['allocated'][sim_ids_to_send] = True

            if block_others:
                unassigned_workers = set(W['worker_id'][W['active']==0]) - set(Work.keys()) - blocked_set
                workers_to_block = list(unassigned_workers)[:np.max(H[sim_ids_to_send]['num_nodes'])-1]
                Work[i]['libE_info']['blocking'] = workers_to_block

        else:
            # Since there is no sim work to give, give gen work. 

            # Limit number of gen instances if given
            if 'num_inst' in gen_specs and gen_count >= gen_specs['num_inst']:
                break

            # Don't give out any gen instances if in batch mode and any point has not been returned or paused
            if 'batch_mode' in gen_specs and gen_specs['batch_mode'] and np.any(np.logical_and(~H['returned'],~H['paused'])):
                break

            # Give gen work 
            gen_count += 1 

            Work[i] = {'persis_info': persis_info[i],
                       'H_fields': gen_specs['in'],
                       'tag': EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': range(0,len(H)),
                                }
                       }

    return Work, persis_info

