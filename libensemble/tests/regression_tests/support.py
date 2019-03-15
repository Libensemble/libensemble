import numpy as np
import copy

def save_libE_output(H,calling_file,nworkers):
    import os
    script_name = os.path.splitext(os.path.basename(calling_file))[0]
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) \
                          + '_evals=' + str(sum(H['returned'])) \
                          + '_ranks=' + str(nworkers)

    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)

##### sim_f #####
# float_x1000
from libensemble.sim_funcs.comms_testing import float_x1000
array_size = int(1e6)   # Size of large array in sim_specs
float_x1000_sim_specs = {'sim_f': float_x1000, # This is the function whose output is being minimized
             'in': ['x'],           # These keys will be given to the above function
             'out': [
                     ('arr_vals',float,array_size),
                     ('scal_val',float),
                    ],
             }

# chwirut1 sim_f
from libensemble.sim_funcs.chwirut1 import chwirut_eval
chwirut_one_at_a_time_sim_specs = {'sim_f': chwirut_eval,
             'in': ['x', 'obj_component'],
             'out': [('f_i',float)],
             }

chwirut_all_sim_specs = {'sim_f': chwirut_eval,
             'in': ['x'],
             'out': [('f',float), ('fvec',float,214),
                     ],
             'combine_component_func': lambda x: np.sum(np.power(x,2)),
             }

# branin sim_f 
from libensemble.sim_funcs.branin.branin_obj import call_branin 

import pkg_resources; branin_sim_dir_name=pkg_resources.resource_filename('libensemble.sim_funcs.branin', '')
branin_sim_specs = {'sim_f': call_branin, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'sim_dir': branin_sim_dir_name, # to be copied by each worker
             'clean_jobs': True,
             }

branin_vals_and_minima =  np.array([[-3.14159, 12.275, 0.397887],
                                    [3.14159, 2.275, 0.397887],
                                    [9.42478, 2.475, 0.397887]])

# six_hump_camel sim_f
from libensemble.sim_funcs.six_hump_camel import six_hump_camel
six_hump_camel_sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             }

from libensemble.sim_funcs.six_hump_camel import six_hump_camel_with_different_ranks_and_nodes
six_hump_camel_with_different_ranks_and_nodes_sim_specs = {'sim_f': six_hump_camel_with_different_ranks_and_nodes, # This is the function whose output is being minimized
             'in': ['x','num_nodes','ranks_per_node'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             }

from libensemble.sim_funcs.six_hump_camel import six_hump_camel_simple
six_hump_camel_simple_sim_specs = copy.deepcopy(six_hump_camel_sim_specs)
six_hump_camel_simple_sim_specs['sim_f'] = six_hump_camel_simple

six_hump_camel_minima = np.array([[ -0.089842,  0.712656],
                       [  0.089842, -0.712656],
                       [ -1.70361,  0.796084],
                       [  1.70361, -0.796084],
                       [ -1.6071,   -0.568651],
                       [  1.6071,    0.568651]])


# one_d_example sim_f
from libensemble.sim_funcs.one_d_func import one_d_example 
one_d_example_sim_specs = copy.deepcopy(six_hump_camel_sim_specs)
one_d_example_sim_specs['sim_f'] = one_d_example

# nan_func sim_f
def nan_func(calc_in,persis_info,sim_specs,libE_info):
    H = np.zeros(1,dtype=sim_specs['out'])
    H['f_i'] = np.nan
    H['f'] = np.nan
    return (H, persis_info)

nan_func_sim_specs = {'sim_f': nan_func, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float),('f_i',float), # This is the output from the function being minimized
                    ],
             }

# job_control_hworld sim_f
from libensemble.sim_funcs.job_control_hworld import job_control_hworld
job_control_hworld_sim_specs = {'sim_f': job_control_hworld, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                     ('cstat',int),
                    ],
             'save_every_k': 400,
}

##### gen_f #####
uniform_or_localopt_gen_out = [
      ('priority',float),
      ('local_pt',bool),
      ('known_to_aposmm',bool), # Mark known points so fewer updates are needed.
      ('dist_to_unit_bounds',float),
      ('dist_to_better_l',float),
      ('dist_to_better_s',float),
      ('ind_of_better_l',int),
      ('ind_of_better_s',int),
      ('started_run',bool),
      ('num_active_runs',int),
      ('local_min',bool),
      ]

# uniform_or_localopt gen_f
from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt
uniform_or_localopt_gen_specs = {'gen_f': uniform_or_localopt,
             'in': [],
             'localopt_method':'LN_BOBYQA',
             'xtol_rel':1e-4,
             'out': uniform_or_localopt_gen_out,
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': 2,
             'batch_mode': True,
             'num_active_gens':1,
             }

# uniform_random_sample gen_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample
uniform_random_sample_gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             }

# persistent_uniform_sampling gen_f
from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform 
persistent_uniform_sampling_gen_specs = copy.deepcopy(uniform_random_sample_gen_specs)
persistent_uniform_sampling_gen_specs['in'] = []
persistent_uniform_sampling_gen_specs['gen_batch_size'] = 20
persistent_uniform_sampling_gen_specs['gen_f'] = persistent_uniform

# uniform_random_sample_obj_components gen_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample_obj_components
uniform_random_sample_obj_components_gen_specs = {'gen_f': uniform_random_sample_obj_components,
             'in': ['pt_id'],
             'out': [('priority',float),
                      ('paused',bool),
                      ('obj_component',int),
                      ('pt_id',int),],
             'gen_batch_size': 2,
             'single_component_at_a_time': True,
             'combine_component_func': lambda x: np.sum(np.power(x,2)),
             'num_active_gens': 1,
             'batch_mode': True,
             }

# aposmm gen_f
aposmm_gen_out = copy.deepcopy(uniform_or_localopt_gen_out) 
aposmm_gen_out += [('sim_id',int),
                   ('paused',bool),
                   ('pt_id',int), # Used to identify points evaluated by different simulations
                   ]

# uniform_random_sample_with_different_nodes_and_ranks gen_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample_with_different_nodes_and_ranks
uniform_random_sample_with_different_nodes_and_ranks_gen_specs = \
        {'gen_f': uniform_random_sample_with_different_nodes_and_ranks,
             'in': ['sim_id'],
             'out': [('priority',float),
                     ('num_nodes',int),
                     ('ranks_per_node',int),
                    ],
             'initial_batch_size': 5,
             'max_ranks_per_node': 8,
             'num_active_gens': 1,
             'batch_mode': False,
             'give_all_with_same_priority': True,
             }

from libensemble.gen_funcs.aposmm import aposmm_logic
aposmm_without_grad_gen_specs = {'gen_f': aposmm_logic,
             'in': [o[0] for o in aposmm_gen_out] + ['f', 'returned'],
             'out': aposmm_gen_out,
             'initial_sample_size': 5,
             'num_active_gens': 1,
             'batch_mode': True,
             }

aposmm_with_grad_gen_specs = copy.deepcopy(aposmm_without_grad_gen_specs) 
aposmm_with_grad_gen_specs['in'] += ['grad']

##### alloc_f #####
# only_persistent_gens alloc_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens 
only_persistent_gens_alloc_specs = {'out':[], 'alloc_f':only_persistent_gens}

# start_persistent_local_opt_gens alloc_f #####
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens
start_persistent_local_opt_gens_alloc_specs = {'out':uniform_or_localopt_gen_out, 'alloc_f':start_persistent_local_opt_gens}

# give_sim_work_first_aposmm alloc_f 
from libensemble.alloc_funcs.fast_alloc_to_aposmm import give_sim_work_first as give_sim_work_first_apossm
give_sim_work_first_aposmm_alloc_specs = {'out':[('allocated',bool)], 'alloc_f':give_sim_work_first_apossm}

# give_sim_work_first alloc_f 
from libensemble.alloc_funcs.fast_alloc import give_sim_work_first  
give_sim_work_first_alloc_specs = {'out':[('allocated',bool)], 'alloc_f':give_sim_work_first}

# give_sim_work_first_pausing alloc_f 
from libensemble.alloc_funcs.fast_alloc_and_pausing import give_sim_work_first  as give_sim_work_first_pausing
give_sim_work_first_pausing_alloc_specs = {'out':[('allocated',bool)], 
               'alloc_f':give_sim_work_first_pausing,
               'stop_on_NaNs': True,
               'stop_partial_fvec_eval': True,
               }

##### persis_info #####

def give_each_worker_own_stream(persis_info,nworkers):
    for i in range(nworkers):
        if i in persis_info:
            persis_info[i].update({'rand_stream': np.random.RandomState(i), 'worker_num': i})
        else:
            persis_info[i] = {'rand_stream': np.random.RandomState(i), 'worker_num': i}
    return persis_info

# give_sim_work_first persis_info 
persis_info_1={}
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
