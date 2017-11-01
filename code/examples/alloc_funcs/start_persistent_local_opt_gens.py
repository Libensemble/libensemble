from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from message_numbers import EVAL_SIM_TAG 
from message_numbers import EVAL_GEN_TAG 

sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
import aposmm_logic 

def start_persistent_local_opt_gens(active_w, idle_w, persis_w, H, H_ind, sim_specs, gen_specs, term_test, gen_info):
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

    if len(gen_info) == 0: 
        gen_info[0] = {}
        gen_info[0]['rand_stream'] = {i:np.random.RandomState(i) for i in idle_w}

    for i in idle_w:
        if term_test(H, H_ind):
            break

        # Find candidate points for starting local opt runs
        n, n_s, c_flag, _, rk_const, lhs_divisions, mu, nu = initialize_APOSMM(H, gen_specs)
        update_history_dist(H, gen_specs, c_flag=False)
        starting_inds = decide_where_to_start_localopt(H, n_s, rk_const, lhs_divisions, mu, nu)        

        # Start up a persistent generator that is a local opt run if all workers won't be persistent generators.
        if gen_count + len(active_w[EVAL_GEN_TAG] + persis_w[PERSIS_GEN_TAG]) <= len(idle_w + active_w[EVAL_SIM_TAG]):
            # Start at the best possible starting point 
            ind = starting_inds[np.argmin(H['f'][starting_inds])]


            Work[i] = {'gen_info':gen_info[i],
                       'H_fields': gen_specs['in'],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': ind,
                                     'gen_num': i
                                }
                       }

            H['started_run'][ind] = 1
            H['num_active_runs'][ind] += 1

            persis_w['PERSIS_GEN_TAG'].add(i)


        else: 
            # Else, perform sim evaluations from existing runs

            # Find indices of H where that are not given nor paused
            q_inds_logical = np.logical_and(~H['given'][:H_ind],~H['paused'][:H_ind])

            if np.any(q_inds_logical):
                sim_ids_to_send = np.nonzero(q_inds_logical)[0][0] # oldest point

                Work[i] = {'H_fields': sim_specs['in'],
                           'gen_info': {}, # Our sims don't need information about how points were generatored
                           'tag':EVAL_SIM_TAG, 
                           'libE_info': {'H_rows': sim_ids_to_send,
                                    },
                          }

            else:
                # There are no points available, so we call our gen_func
                Work[i] = {'gen_info':gen_info[0],
                           'H_fields': gen_specs['in'],
                           'tag':EVAL_GEN_TAG, 
                           'libE_info': {'H_rows': range(0,H_ind),
                                         'gen_num': 0
                                    }
                           }

    return Work, gen_info

