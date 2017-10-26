from __future__ import division
from __future__ import absolute_import
import numpy as np
import time
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from message_numbers import EVAL_SIM_TAG 
from message_numbers import EVAL_GEN_TAG 

def give_sim_work_first(active_w, idle_w, H, H_ind, sim_specs, gen_specs, term_test, gen_info):
    """ Decide what should be given to workers. Note that everything put into
    the Work dictionary will be given, so we are careful not to put more gen or
    sim items into Work than necessary.
    """

    Work = {}
    gen_count = 0

    if len(gen_info) == 0: 
        gen_info[0] = {}

    for i in idle_w:
        if term_test(H, H_ind):
            break

        # Only consider giving to worker i if it's resources are not blocked by some other calculation
        blocked_set = active_w['blocked'].union(*[j['libE_info']['blocking'] for j in Work.values() if 'blocking' in j['libE_info']])
        if i in blocked_set:
            continue

        # Find indices of H where that are not given nor paused
        q_inds_logical = np.logical_and(~H['given'][:H_ind],~H['paused'][:H_ind])

        if np.any(q_inds_logical):
            # Give sim work if possible

            if 'priority' in H.dtype.fields:
                if 'give_all_with_same_priority' in gen_specs and gen_specs['give_all_with_same_priority']:
                    # Give all points with highest priority
                    q_inds = H['priority'][:H_ind][q_inds_logical] == np.max(H['priority'][:H_ind][q_inds_logical])
                    sim_ids_to_send = np.nonzero(q_inds_logical)[0][q_inds]
                else:
                    # Give first point with highest priority
                    sim_ids_to_send = np.nonzero(q_inds_logical)[0][np.argmax(H['priority'][:H_ind][q_inds_logical])]
            else:
                # Give oldest point
                sim_ids_to_send = np.nonzero(q_inds_logical)[0][0]

            sim_ids_to_send = np.atleast_1d(sim_ids_to_send)

            # Only give work if enough idle workers
            if 'num_nodes' in H.dtype.names and np.any(H[sim_ids_to_send]['num_nodes'] > 1):
                if np.any(H[sim_ids_to_send]['num_nodes'] > len(idle_w) - len(Work) - len(blocked_set)):
                    # Worker i doesn't get any work. Just waiting for other resources to open up
                    continue
                block_others = True
            else:
                block_others = False

            Work[i] = {'H_fields': sim_specs['in'],
                       'gen_info': {}, # Our sims don't need information about how points were generatored
                       'tag':EVAL_SIM_TAG, 
                       'libE_info': {'H_rows': sim_ids_to_send,
                                },
                      }

            if block_others:
                unassigned_workers = idle_w - set(Work.keys()) - blocked_set
                workers_to_block = list(unassigned_workers)[:np.max(H[sim_ids_to_send]['num_nodes'])-1]
                Work[i]['libE_info']['blocking'] = set(workers_to_block)

            update_history_x_out(H, sim_ids_to_send, i)

        else:
            # Since there is no sim work to give, give gen work. 

            # Limit number of gen instances if given
            if 'num_inst' in gen_specs and len(active_w[EVAL_GEN_TAG]) + gen_count >= gen_specs['num_inst']:
                break

            # Don't give out any gen instances if in batch mode and any point has not been returned or paused
            if 'batch_mode' in gen_specs and gen_specs['batch_mode'] and np.any(np.logical_and(~H['returned'][:H_ind],~H['paused'][:H_ind])):
                break

            # Give gen work 
            gen_count += 1 

            Work[i] = {'gen_info':gen_info[0],
                       'H_fields': gen_specs['in'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': range(0,H_ind),
                                     'gen_num': 0
                                }
                       }

    return Work, gen_info



def update_history_x_out(H, q_inds, lead_rank):
    """
    Updates the history (in place) when a new point has been given out to be evaluated

    Parameters
    ----------
    H: numpy structured array
        History array storing rows for each point.
    H_ind: integer
        The new point
    W: numpy array
        Work to be evaluated
    lead_rank: int
        lead ranks for the evaluation of x 
    """

    for i,j in zip(q_inds,range(len(q_inds))):
        # for field in W.dtype.names:
        #     H[field][i] = W[field][j]

        H['given'][i] = True
        H['given_time'][i] = time.time()
        H['lead_rank'][i] = lead_rank
