import numpy as np
import copy

branin_vals_and_minima = np.array([[-3.14159, 12.275, 0.397887],
                                   [3.14159, 2.275, 0.397887],
                                   [9.42478, 2.475, 0.397887]])

six_hump_camel_minima = np.array([[-0.089842, 0.712656], [0.089842, -0.712656],
                                  [-1.70361, 0.796084], [1.70361, -0.796084],
                                  [-1.6071, -0.568651], [1.6071, 0.568651]])


def nan_func(calc_in, persis_info, sim_specs, libE_info):
    H = np.zeros(1, dtype=sim_specs['out'])
    H['f_i'] = np.nan
    H['f'] = np.nan
    return (H, persis_info)


uniform_or_localopt_gen_out = [('priority', float),
                               ('local_pt', bool),
                               ('known_to_aposmm', bool),
                               ('dist_to_unit_bounds', float),
                               ('dist_to_better_l', float),
                               ('dist_to_better_s', float),
                               ('ind_of_better_l', int),
                               ('ind_of_better_s', int),
                               ('started_run', bool),
                               ('num_active_runs', int),
                               ('local_min', bool)]

aposmm_gen_out = copy.deepcopy(uniform_or_localopt_gen_out)
aposmm_gen_out += [('sim_id', int),
                   ('paused', bool),
                   ('pt_id', int)]  # Identify the same point evaluated by different sim_f's or components

# give_sim_work_first persis_info
persis_info_1 = {}
# Below persis_info fields store APOSMM information, but can be passed to various workers.
persis_info_1['total_gen_calls'] = 0
persis_info_1['last_worker'] = 0
persis_info_1['next_to_give'] = 0
persis_info_1[0] = {'run_order': {},
                    'old_runs': {},
                    'total_runs': 0,
                    'rand_stream': np.random.RandomState(1)}

persis_info_2 = copy.deepcopy(persis_info_1)
persis_info_2[1] = persis_info_2[0]
persis_info_2.pop(0)

# give_sim_work_first_pausing persis_info
persis_info_3 = copy.deepcopy(persis_info_1)
persis_info_3.pop('next_to_give')
persis_info_3['need_to_give'] = set()
persis_info_3['complete'] = set()
persis_info_3['has_nan'] = set()
persis_info_3['already_paused'] = set()
persis_info_3['H_len'] = 0
persis_info_3['best_complete_val'] = np.inf
persis_info_3['local_pt_ids'] = set()
persis_info_3['inds_of_pt_ids'] = {}
