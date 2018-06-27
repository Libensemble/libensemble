from __future__ import division
from __future__ import absolute_import
import numpy as np
import sys, os

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from libensemble.message_numbers import EVAL_SIM_TAG 
from libensemble.message_numbers import EVAL_GEN_TAG 

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
import libensemble.gen_funcs.aposmm_logic as aposmm_logic

def start_persistent_local_opt_gens(worker_sets, H, sim_specs, gen_specs, gen_info):
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
    already_in_Work = np.zeros(len(H),dtype=bool) # To mark points as they are included in Work, but not yet marked as 'given' in H.

    # At startup, build an intial gen_info
    if len(gen_info) == 0: 
        gen_info = {}
        for i in worker_sets['nonpersis_w']['waiting']:
            gen_info[i] = {'rand_stream': np.random.RandomState(i)}

    # If a persistent localopt run has just finished, use run_order to update H
    # and then remove other information from gen_info
    for i in gen_info.keys():
        if 'done' in gen_info[i]:
            H['num_active_runs'][gen_info[i]['run_order']] -= 1
        if 'x_opt' in gen_info[i]:
            opt_ind = np.all(H['x']==gen_info[i]['x_opt'],axis=1)
            assert sum(opt_ind) == 1, "There must be just one optimum"
            H['local_min'][opt_ind] = True
            gen_info[i] = {'rand_stream': gen_info[i]['rand_stream']}

    # If i is idle, but in persistent mode, and its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in worker_sets['persis_w']['waiting'][EVAL_GEN_TAG]: 
        gen_inds = H['gen_worker']==i 
        if np.all(H['returned'][gen_inds]):
            last_ind = np.nonzero(gen_inds)[0][np.argmax(H['given_time'][gen_inds])]
            Work[i] = {'gen_info':gen_info[i],
                       'H_fields': sim_specs['in'] + [name[0] for name in sim_specs['out']],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(last_ind),
                                     'gen_num': i,
                                     'persistent': True
                                }
                       }
            gen_info[i]['run_order'].append(last_ind)

    for i in worker_sets['nonpersis_w']['waiting']:
        # Find candidate points for starting local opt runs if a sample point has been evaluated
        if np.any(np.logical_and(~H['local_pt'],H['returned'])):
            n, n_s, c_flag, _, rk_const, lhs_divisions, mu, nu = aposmm_logic.initialize_APOSMM(H, gen_specs)
            aposmm_logic.update_history_dist(H, gen_specs, c_flag=False)
            starting_inds = aposmm_logic.decide_where_to_start_localopt(H, n_s, rk_const, lhs_divisions, mu, nu)        
        else:
            starting_inds = []

        # Start up a persistent generator that is a local opt run but don't do it if all workers will be persistent generators.
        if len(starting_inds) and gen_count + len(worker_sets['persis_w'][EVAL_GEN_TAG]) + 1 < len(worker_sets['nonpersis_w']['waiting']) + len(worker_sets['nonpersis_w'][EVAL_GEN_TAG]) + len(worker_sets['nonpersis_w'][EVAL_SIM_TAG]): 
            # Start at the best possible starting point 
            ind = starting_inds[np.argmin(H['f'][starting_inds])]

            Work[i] = {'gen_info':gen_info[i],
                       'H_fields': sim_specs['in'] + [name[0] for name in sim_specs['out']],
                       'tag':EVAL_GEN_TAG, 
                       'libE_info': {'H_rows': np.atleast_1d(ind),
                                     'gen_num': i,
                                     'persistent': True
                                }
                       }

            H['started_run'][ind] = 1
            H['num_active_runs'][ind] += 1

            gen_info[i]['run_order'] = [ind]
            gen_count += 1

        else: 
            # Else, perform sim evaluations from existing runs (if they exist).
            q_inds_logical = np.logical_and.reduce((~H['given'],~H['paused'],~already_in_Work))

            if np.any(q_inds_logical):
                b = np.logical_and(q_inds_logical,  H['local_pt'])
                if np.any(b):
                    q_inds_logical = b
                else:
                    q_inds_logical = np.logical_and(q_inds_logical, ~H['local_pt'])

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
                # Finally, generate points since there is nothing else to do. 
                if gen_count + len(worker_sets['nonpersis_w'][EVAL_GEN_TAG] | worker_sets['persis_w']['waiting'][EVAL_GEN_TAG] | worker_sets['persis_w'][EVAL_GEN_TAG]) > 0: 
                    continue
                gen_count += 1
                # There are no points available, so we call our gen_func
                Work[i] = {'gen_info':gen_info[i],
                           'H_fields': gen_specs['in'],
                           'tag':EVAL_GEN_TAG, 
                           'libE_info': {'H_rows': [],
                                         'gen_num': i
                                    }
                           }

    return Work, gen_info

