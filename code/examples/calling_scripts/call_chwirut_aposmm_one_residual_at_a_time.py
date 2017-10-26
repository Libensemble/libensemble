# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command: 

# mpiexec -np 4 python3 call_chwirut_aposmm_one_residual_at_a_time.py

# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
import numpy as np

sys.path.append('../../src')
from libE import libE

sys.path.append('../sim_funcs')
sys.path.append('../gen_funcs')
from chwirut1 import sum_squares, libE_func_wrapper
from aposmm_logic import aposmm_logic, queue_update_function
from math import gamma, sqrt, pi

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 10*m

sim_specs = {'sim_f': [libE_func_wrapper],
             'in': ['x', 'obj_component'],
             'out': [('f_i',float),
                     ],
             }

out = [('x',float,n),
      ('x_on_cube',float,n),
      ('sim_id',int),
      ('priority',float),
      ('local_pt',bool),
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
             'in': [o[0] for o in out] + ['f_i', 'returned'],
             'out': out,
             'lb': -2*np.ones(3),
             'ub':  2*np.ones(3),
             'initial_sample': 10, # All 214 residuals must be done
             'localopt_method': 'LN_BOBYQA',
             'localopt_method': 'pounders',
             'delta_0_mult': 0.5,
             'grtol': 1e-4,
             'gatol': 1e-4,
             'frtol': 1e-15,
             'fatol': 1e-15,
             'rk_const': ((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
             'xtol_rel': 1e-3,
             'min_batch_size': 1,
             # 'min_batch_size': 20*len(alloc_specs['worker_ranks']),
             'single_component_at_a_time': True,
             'components': m,
             'combine_component_func': sum_squares,
             'num_inst': 1,
             'batch_mode': True,
             'stop_on_NaNs': True, 
             'stop_partial_fvec_eval': True,
             'queue_update_function': queue_update_function 
             }

exit_criteria = {'sim_max': max_sim_budget, # must be provided
                  }

np.random.seed(1)
# Perform the run
H = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    filename = 'chwirut_results_after_evals=' + str(max_sim_budget) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)
