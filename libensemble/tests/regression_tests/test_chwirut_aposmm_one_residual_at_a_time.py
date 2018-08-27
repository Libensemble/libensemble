# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command:

# mpiexec -np 4 python3 call_chwirut_aposmm_one_residual_at_a_time.py

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
             }

gen_out = [('x',float,n),
      ('x_on_cube',float,n),
      ('sim_id',int),
      ('priority',float),
      ('local_pt',bool),
      ('paused',bool),
      ('known_to_aposmm',bool), # Mark known points so fewer updates are needed.
      ('dist_to_unit_bounds',float),
      ('dist_to_better_l',float),
      ('dist_to_better_s',float),
      ('ind_of_better_l',int),
      ('ind_of_better_s',int),
      ('started_run',bool),
      ('num_active_runs',int), # Number of active runs point is involved in
      ('local_min',bool),
      ('obj_component',int),
      ('f',float), # To store the point's combined objective function value (after all f_i are computed)
      ('pt_id',int), # To be used by APOSMM to identify points evaluated by different simulations
      ]

gen_specs = {'gen_f': aposmm_logic,
             'in': [o[0] for o in gen_out] + ['f_i','returned'],
             'out': gen_out,
             'lb': -2*np.ones(3),
             'ub':  2*np.ones(3),
             'initial_sample_size': 5, # All 214 residuals must be done
             'localopt_method': 'pounders',
             'dist_to_bound_multiple': 0.5,
             'grtol': 1e-4,
             'gatol': 1e-4,
             'frtol': 1e-15,
             'fatol': 1e-15,
             'single_component_at_a_time': True,
             'components': m,
             'combine_component_func': lambda x: np.sum(np.power(x,2)),
             'num_active_gens': 1,
             'batch_mode': True,
             'stop_on_NaNs': True,
             'stop_partial_fvec_eval': True,
             }

np.random.RandomState(0)
gen_specs['sample_points'] = np.random.uniform(0,1,(max_sim_budget,n))*(gen_specs['ub']-gen_specs['lb'])+gen_specs['lb']

exit_criteria = {'sim_max': max_sim_budget, # must be provided
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
# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert len(H) >= max_sim_budget
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_after_evals=' + str(max_sim_budget) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)
