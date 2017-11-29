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
from chwirut1 import libE_func_wrapper, EvaluateJacobian
 
# Import gen_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from aposmm_logic import aposmm_logic, queue_update_function

script_name = os.path.splitext(os.path.basename(__file__))[0]

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 10

sim_specs = {'sim_f': [libE_func_wrapper],
             'in': ['x'],
             'out': [('f',float), ('fvec',float,m),
                     ],
             'combine_component_func': lambda x: np.sum(np.power(x,2)),
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
      ('pt_id',int), # To be used by APOSMM to identify points evaluated by different simulations
      ]

gen_specs = {'gen_f': aposmm_logic,
             'in': [o[0] for o in gen_out] + ['f', 'fvec', 'returned'],
             'out': gen_out,
             'lb': -2*np.ones(3),
             'ub':  2*np.ones(3),
             'initial_sample': 5, # All 214 residuals must be done
             'localopt_method': 'pounders',
             'delta_0_mult': 0.5,
             'grtol': 1e-4,
             'gatol': 1e-4,
             'frtol': 1e-15,
             'fatol': 1e-15,
             'components': m,
             'num_inst': 1,
             'batch_mode': True,
             'queue_update_function': queue_update_function 
             }

exit_criteria = {'sim_max': max_sim_budget, # must be provided
                  }

np.random.seed(1)
# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert len(H) >= max_sim_budget
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_after_evals=' + str(max_sim_budget) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)

    # Calculating the Jacobian at the best point (though this information was not used by pounders)
    J = EvaluateJacobian(H['x'][np.argmin(H['f'])])
    assert np.linalg.norm(J) < 2000

