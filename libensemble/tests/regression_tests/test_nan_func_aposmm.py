# """
# Runs libEnsemble on a function that returns only nan; tests APOSMM functionality 
# 
# Execute via the following command:
#    mpiexec -np 4 python3 test_nan_func_aposmm.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
#sys.path.append('../../src')
from libensemble.libE import libE

# Import gen_func 
from libensemble.gen_funcs.aposmm_logic import aposmm_logic
from math import gamma, pi, sqrt

script_name = os.path.splitext(os.path.basename(__file__))[0]

n = 2

def nan_func(calc_in,gen_info,sim_specs,libE_info):
    H = np.zeros(1,dtype=sim_specs['out'])
    H['f_i'] = np.nan
    H['f'] = np.nan
    return (H, gen_info)

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': nan_func, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float),('f_i',float),('grad',float,n), # This is the output from the function being minimized
                    ],
             }

gen_out = [('x',float,n),
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
      ('pt_id',int), # To be used by APOSMM to identify points evaluated by different simulations
      ]

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': aposmm_logic,
             'in': [o[0] for o in gen_out] + ['f','f_i', 'grad', 'returned'],
             'out': gen_out,
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'initial_sample': 100,
             'localopt_method': 'LD_MMA',
             'rk_const': 0.5*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
             'xtol_rel': 1e-2,
             'lhs_divisions':2,
             'batch_mode': True,
             'num_inst':1,
             }

w = MPI.COMM_WORLD.Get_size()-1
if w == 3:
    gen_specs['single_component_at_a_time'] = True
    gen_specs['components'] = 1
    gen_specs['combine_component_func'] = np.linalg.norm

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 1000}

np.random.seed(1)

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria)
if MPI.COMM_WORLD.Get_rank() == 0:
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)

    assert np.all(~H['local_pt']) 



