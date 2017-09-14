# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. (You will need to run "make gkls_single" in libensemble/code/examples/sim_funcs/GKLS/GKLS_sim_src/
# before running this script with 

# mpiexec -np 4 python3 call_GKLS_aposmm.py

# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
import os    
import numpy as np

sys.path.append('../../src')
from libE import libE

# Declare the objective
GKLS_dir_name='../sim_funcs/GKLS/GKLS_sim_src'
sys.path.append(GKLS_dir_name)
from GKLS_obj import call_GKLS as obj_func

sys.path.append('../sim_funcs')
sys.path.append('../gen_funcs')
from chwirut1 import sum_squares
from aposmm_logic import aposmm_logic

from math import gamma, sqrt, pi

### Declare the run parameters/functions
max_sim_budget = 600
n = 2
w = MPI.COMM_WORLD.Get_size()-1

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': [obj_func], # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'params': {'number_of_minima': 10, # These are parameters needed by the function being minimized.
                        'problem_dimension': 2,
                        'problem_number': 1,
                        # 'sim_dir': './GKLS_sim_src'}, # to be copied by each worker 
                        'sim_dir': GKLS_dir_name}, # to be copied by each worker 
             }


out = [('x',float,n),
      ('x_on_cube',float,n),
      ('sim_id',int),
      ('priority',float),
      ('iter_plus_1_in_run_id',int,max_sim_budget),
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
      ]

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': aposmm_logic,
             'in': [o[0] for o in out] + ['f', 'returned'],
             'out': out,
             'params': {'lb': np.array([0,0]),
                        'ub': np.array([1,1]),
                        'initial_sample': 40,
                        'localopt_method': 'LN_BOBYQA',
                        # 'localopt_method': 'pounders',
                        # 'delta_0_mult': 0.5,
                        # 'grtol': 1e-4,
                        # 'gatol': 1e-4,
                        # 'frtol': 1e-15,
                        # 'fatol': 1e-15,
                        'rk_const': ((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
                        'xtol_rel': 1e-3,
                        'min_batch_size': w,
                       },
             'num_inst': 1,
             'batch_mode': True,
             }

# Tell LibEnsemble when to stop
exit_criteria = {'sim_max': max_sim_budget, 
                 'elapsed_wallclock_time': 100,
                 'stop_val': ('f', -1), # key must be in sim_specs['out'] or gen_specs['out'] 
                }

np.random.seed(1)
# Perform the run

# H0 = np.load('GKLS_results_after_evals=500_ranks=2.npy')
# H0 = H0[['x','x_on_cube','f']][:50]

H = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    filename = 'GKLS_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(w)
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)

    minima_and_func_val_file = os.path.join(GKLS_dir_name, 'which_seeds_are_feasible/known_minima_and_func_values_for_n=' + str(sim_specs['params']['problem_dimension']) + '_prob=' + str(sim_specs['params']['problem_number']) + '_min=' + str(sim_specs['params']['number_of_minima']))

    if os.path.isfile(minima_and_func_val_file):
        M = np.loadtxt(minima_and_func_val_file)
        M = M[M[:,-1].argsort()] # Sort by function values (last column)
        k = 4
        tol = 1e-7
        for i in range(k):
            assert np.min(np.sum((H['x'][H['local_min']]-M[i,:n])**2,1)) < tol

        print("\nLibEnsemble with APOSMM has identified the " + str(k) + " best minima within a tolerance " + str(tol))
