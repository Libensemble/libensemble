# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command:

# mpiexec -np 4 python3 test_chwirut_uniform_sampling_one_residual_at_a_time.py

# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
from libensemble.libE import libE

# Import sim_func
from libensemble.sim_funcs.chwirut1 import chwirut_eval

# Import gen_func
from libensemble.gen_funcs.aposmm import aposmm_logic, queue_update_function
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample_obj_components

# Import alloc_func
from libensemble.alloc_funcs.fast_alloc_to_aposmm import give_sim_work_first as alloc_f

script_name = os.path.splitext(os.path.basename(__file__))[0]

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 10*m

sim_specs = {'sim_f': chwirut_eval,
             'in': ['x', 'obj_component'],
             'out': [('f_i',float),
                     ],
             'component_nan_frequency': 0.05,
             }

gen_out = [('x',float,n),
      ('priority',float),
      ('paused',bool),
      ('obj_component',int),
      ('pt_id',int),
      ]

gen_specs = {'gen_f': uniform_random_sample_obj_components,
             'in': ['pt_id'],
             'out': gen_out,
             'lb': -2*np.ones(3),
             'ub':  2*np.ones(3),
             'gen_batch_size': 2,
             'single_component_at_a_time': True,
             'components': m,
             'combine_component_func': lambda x: np.sum(np.power(x,2)),
             'num_active_gens': 1,
             'batch_mode': True,
             'stop_on_NaNs': True,
             'stop_partial_fvec_eval': True,
             }

exit_criteria = {'sim_max': max_sim_budget, # must be provided
                 'elapsed_wallclock_time': 300
                  }

alloc_specs = {'out':[('allocated',bool)], 'alloc_f':alloc_f}

libE_specs = {'queue_update_function': queue_update_function}
np.random.seed(1)
persis_info = {'next_to_give':0}
persis_info['total_gen_calls'] = 0
persis_info['complete'] = set()
persis_info['has_nan'] = set()
persis_info['already_paused'] = set()
persis_info['H_len'] = 0

for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}

persis_info['last_worker'] = 0
persis_info[0] = {'active_runs': set(),
                  'run_order': {},
                  'old_runs': {},
                  'total_runs': 0,
                  'rand_stream': np.random.RandomState(1)}
# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_after_evals=' + str(max_sim_budget) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)
