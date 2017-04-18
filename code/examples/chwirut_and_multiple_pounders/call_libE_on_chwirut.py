# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. (You will need to run "make gkls_single" in libensemble/examples/GKLS/
# before running this script with 

# mpiexec -np 4 python3 call_libE_on_GKLS.py

# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
sys.path.append('../../src')

import numpy as np
from libE import libE
from chwirut1 import EvaluateFunction, EvaluateJacobian
from aposmm_logic import aposmm_logic
from math import *

def sum_squares(x):
    return np.sum(np.power(x,2))

def libE_func_wrapper(H,sim_out,params):

    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_out)

    for i,x in enumerate(H['x']):
        O['fvec'][i] = EvaluateFunction(x)
        O['f'] = params['combine_component_func'](O['fvec'][i])

    return O
        

### Declare the run parameters/functions
n = 3
max_evals = 5
c = {}
c['comm'] = MPI.COMM_WORLD
c['color'] = 0

allocation_specs = {'manager_ranks': set([0]), 
                    'worker_ranks': set(range(1,c['comm'].Get_size()))
                   }

sim_specs = {'f': [libE_func_wrapper],
             'in': ['x'],
             'out': [('fvec','float',214),
                     ('f','float'),
                     # ('Jacobian','float',(214,n)),
                     ],
             'params': {'combine_component_func': sum_squares,
                        }, 
             }

gen_specs = {'f': aposmm_logic,
             'in': ['x_on_cube', 'fvec', 'f', 'local_pt', 'known_to_aposmm', 'iter_plus_1_in_run_id', 'dist_to_unit_bounds',
                    'dist_to_better_l', 'dist_to_better_s', 'ind_of_better_l',
                    'ind_of_better_s', 'started_run', 'num_active_runs', 'local_min','returned','pt_id'],
             'out': [('x','float',n),
                     ('x_on_cube','float',n),
                     ('priority','float'),
                     ('iter_plus_1_in_run_id','int',max_evals),
                     ('local_pt','bool'),
                     ('known_to_aposmm','bool'), # Mark known points so fewer updates are needed.
                     ('dist_to_unit_bounds','float'),
                     ('dist_to_better_l','float'),
                     ('dist_to_better_s','float'),
                     ('ind_of_better_l','int'),
                     ('ind_of_better_s','int'),
                     ('started_run','bool'),
                     ('num_active_runs','int'), # Number of active runs point is involved in
                     ('local_min','bool'),
                     ],
             'params': {'lb': -2*np.ones(3),
                        'ub':  2*np.ones(3),
                        'initial_sample': 2,
                        'localopt_method': 'LN_BOBYQA',
                        'rk_const': ((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
                        'xtol_rel': 1e-3,
                        'num_inst': 1,
                        },
             }

failure_processing = {}

exit_criteria = {'sim_eval_max': max_evals, # must be provided
                  }

np.random.seed(1)
# Perform the run
H = libE(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    np.save('H_after_5_evals',H)
