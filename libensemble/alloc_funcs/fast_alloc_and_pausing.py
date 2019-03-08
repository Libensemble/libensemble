from __future__ import division
from __future__ import absolute_import

import numpy as np
import sys

from libensemble.alloc_funcs.support import \
     avail_worker_ids, sim_work, gen_work, count_gens

def give_sim_work_first(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function gives (in order) entries in ``H`` to idle workers
    to evaluate in the simulation function. The fields in ``sim_specs['in']``
    are given. If all entries in `H` have been given a be evaluated, a worker
    is told to call the generator function, provided this wouldn't result in
    more than ``gen_specs['num_active_gen']`` active generators. Also allows
    for a 'batch_mode'.

    When there are multiple objective components, this allocation function
    does not evaluate further components for some point in the following
    scenarios:

    alloc_specs['stop_on_NaNs']: True --- after a NaN has been found in returned in some
        objective component
    allocated['stop_partial_fvec_eval']: True --- after the value returned from
        combine_component_func is larger than a known upper bound on the objective.

    :See:
        ``/libensemble/tests/regression_tests/test_chwirut_uniform_sampling_one_residual_at_a_time.py``
    """

    Work = {}
    gen_count = count_gens(W)

    if len(H)!=persis_info['H_len']:
        # Something new is in the history.
        persis_info['need_to_give'].update(H['sim_id'][persis_info['H_len']:].tolist())
        persis_info['H_len']=len(H)

    for i in avail_worker_ids(W):

        pt_ids_to_pause = set()

        # Find indices of H that are not yet allocated
        if len(persis_info['need_to_give']):
            # Pause entries in H if one component is evaluated at a time and there are
            # any NaNs for some components.
            if 'stop_on_NaNs' in alloc_specs and alloc_specs['stop_on_NaNs']:
                pt_ids_to_pause.update(H['pt_id'][np.isnan(H['f_i'])])

            # Pause entries in H if a partial combine_component_func evaluation is
            # worse than the best, known, complete evaluation (and the point is not a
            # local_opt point).
            if 'stop_partial_fvec_eval' in alloc_specs and alloc_specs['stop_partial_fvec_eval']:
                pt_ids = np.unique(H['pt_id'])

                for j,pt_id in enumerate(pt_ids):
                    if (pt_id in persis_info['has_nan']) or \
                       (pt_id in persis_info['complete']):
                        continue

                    a1 = H['pt_id']==pt_id
                    if np.any(np.isnan(H['f_i'][a1])):
                        persis_info['has_nan'].add(pt_id)
                        continue

                    if np.all(H['returned'][a1]):
                        persis_info['complete'].add(pt_id)

                if len(persis_info['complete']) and len(pt_ids)>1:
                    complete_fvals_flag = np.zeros(len(pt_ids),dtype=bool)
                    sys.stdout.flush()
                    complete_fvals_flag[list(persis_info['complete'])] = True

                    # Ensure combine_component_func calculates partial fevals correctly
                    # with H['f_i'] = 0 for non-returned point
                    possibly_partial_fvals = np.array([gen_specs['combine_component_func'](H['f_i'][H['pt_id']==j]) for j in pt_ids])

                    best_complete = np.nanmin(possibly_partial_fvals[complete_fvals_flag])

                    worse_flag = np.zeros(len(pt_ids),dtype=bool)
                    for j in range(len(pt_ids)):
                        if not np.isnan(possibly_partial_fvals[j]) and possibly_partial_fvals[j] > best_complete:
                            worse_flag[j] = True

                    # Pause incompete evaluations with worse_flag==True
                    pt_ids_to_pause.update(pt_ids[np.logical_and(worse_flag,~complete_fvals_flag)])

            if not pt_ids_to_pause.issubset(persis_info['already_paused']):
                persis_info['already_paused'].update(pt_ids_to_pause)
                sim_ids_to_remove = np.in1d(H['pt_id'],list(pt_ids_to_pause))
                H['paused'][sim_ids_to_remove] = True

                persis_info['need_to_give'] = persis_info['need_to_give'].difference(np.where(sim_ids_to_remove)[0])

            if len(persis_info['need_to_give']) == 0: 
                import ipdb; ipdb.set_trace()
                continue

            next_row = persis_info['need_to_give'].pop()
            sim_work(Work, i, sim_specs['in'], [next_row], [])

        elif gen_count < gen_specs.get('num_active_gens', gen_count+1):
            lw = persis_info['last_worker']

            last_size = persis_info.get('last_size')
            if len(H): 
                # Don't give gen instances in batch mode if points are unfinished
                if (gen_specs.get('batch_mode')
                    and not all(np.logical_or(H['returned'][last_size:],
                                              H['paused'][last_size:]))):
                    break
                # Don't call APOSMM if there are runs going but none need advancing
                if len(persis_info[lw]['run_order']):
                    runs_needing_to_advance = np.zeros(len(persis_info[lw]['run_order']),dtype=bool)
                    for run,inds in enumerate(persis_info[lw]['run_order'].values()):
                        runs_needing_to_advance[run] = np.all(H['returned'][inds])

                    if not np.any(runs_needing_to_advance):
                        break

            persis_info['last_size'] = len(H)

            # Give gen work
            persis_info['total_gen_calls'] += 1
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info[lw])

            persis_info['last_worker'] = i

    return Work, persis_info
