from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from message_numbers import EVAL_SIM_TAG 
from message_numbers import EVAL_GEN_TAG 
from message_numbers import PERSIS_GEN_TAG 
from message_numbers import PERSIS_SIM_TAG 

sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
import aposmm_logic 

def start_persistent_local_opt_gens(active_w, idle_w, persis_w, H, H_ind, sim_specs, gen_specs, gen_info):
    """ Decide what should be given to workers. Note that everything put into
    the Work dictionary will be given, so we are careful not to put more gen or
    sim items into Work than necessary.


    This allocation function will 
    - Start up a persistent generator that is a local opt run at the first point
      identified by APOSMM's decide_where_to_start_localopt.
    - It will only do this if at least one worker will be left to perform
      simulation evaluations.
    - If multiple starting points are available, the one with smallest function
      value is chosen. 
    - If no candidate starting points exist, points from existing runs will be
      evaluated (oldest first)
    - If no points are left, call the gen_f 
    """

    Work = {}
    gen_count = 0
    already_in_Work = np.zeros(H_ind,dtype=bool) # To mark points as they are included in Work, but not yet marked as 'given' in H.

    if len(gen_info) == 0: 
        gen_info = {}
        for i in idle_w:
            gen_info[i] = {'rand_stream': np.random.RandomState(i)}

    for i in idle_w:

        if i in persis_w[PERSIS_GEN_TAG] | persis_w[PERSIS_SIM_TAG]:
            persis_w['adv'].add(i)


        # Find candidate points for starting local opt runs if a sample point has been evaluated
        if np.any(np.logical_and(~H['local_pt'][:H_ind],H['returned'][:H_ind])):
            n, n_s, c_flag, _, rk_const, lhs_divisions, mu, nu = aposmm_logic.initialize_APOSMM(H[:H_ind], gen_specs)
            aposmm_logic.update_history_dist(H[:H_ind], gen_specs, c_flag=False)
            starting_inds = aposmm_logic.decide_where_to_start_localopt(H[:H_ind], n_s, rk_const, lhs_divisions, mu, nu)        
        else:
            starting_inds = []

        # Start up a persistent generator that is a local opt run if all workers won't be persistent generators.
        if len(starting_inds) and gen_count + len(active_w[EVAL_GEN_TAG]) + len(persis_w[PERSIS_GEN_TAG]) <= len(idle_w) + len(active_w[EVAL_SIM_TAG]):
            # Start at the best possible starting point 
            ind = starting_inds[np.argmin(H['f'][starting_inds])]


            Work[i] = {'gen_info':gen_info[i],
                       'H_fields': ['x'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(ind),
                                     'gen_num': i,
                                     'persistent': True
                                }
                       }

            H['started_run'][ind] = 1
            H['num_active_runs'][ind] += 1

            persis_w[PERSIS_GEN_TAG].add(i)


        else: 
            # Else, perform sim evaluations from existing runs

            # Find indices of H where that are not given nor paused
            q_inds_logical = np.logical_and.reduce((~H['given'][:H_ind],~H['paused'][:H_ind],~already_in_Work))

            if np.any(q_inds_logical):
                sim_ids_to_send = np.nonzero(q_inds_logical)[0][0] # oldest point

                Work[i] = {'H_fields': sim_specs['in'],
                           'gen_info': {}, # Our sims don't need information about how points were generatored
                           'tag':EVAL_SIM_TAG, 
                           'libE_info': {'H_rows': np.atleast_1d(sim_ids_to_send),
                                    },
                          }

                already_in_Work[sim_ids_to_send] = True

            else:
                # There are no points available, so we call our gen_func
                Work[i] = {'gen_info':gen_info[i],
                           'H_fields': gen_specs['in'],
                           'tag':EVAL_GEN_TAG, 
                           'libE_info': {'H_rows': [],
                                         'gen_num': i
                                    }
                           }

    return Work, persis_w, gen_info

