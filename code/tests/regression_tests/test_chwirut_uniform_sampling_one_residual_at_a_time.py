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
sys.path.append('../../src')
from libE import libE

# Import sim_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs'))
from chwirut1 import libE_func_wrapper
 
# Import gen_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from aposmm_logic import queue_update_function
from uniform_sampling import uniform_random_sample_obj_components 

script_name = os.path.splitext(os.path.basename(__file__))[0]

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 10*m

sim_specs = {'sim_f': [libE_func_wrapper],
             'in': ['x', 'obj_component'],
             'out': [('f_i',float),
                     ],
             'component_nan_frequency': 0.05, 
             }

gen_out = [('x',float,n),
      ('priority',float),
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
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
    short_name = script_name.split("test_", 1).pop()    
    filename = short_name + '_results_after_evals=' + str(max_sim_budget) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)
