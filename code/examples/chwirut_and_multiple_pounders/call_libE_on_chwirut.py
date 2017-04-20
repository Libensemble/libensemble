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
from chwirut1 import EvaluateFunction, EvaluateJacobian, sum_squares, libE_func_wrapper
from aposmm_logic import aposmm_logic
from math import *

### Declare the run parameters/functions
n = 3
max_evals = 100
c = {}
c['comm'] = MPI.COMM_WORLD
c['color'] = 0

allocation_specs = {'manager_ranks': set([0]), 
                    'worker_ranks': set(range(1,c['comm'].Get_size()))
                   }

sim_specs = {'f': [libE_func_wrapper],
             'in': ['x'],
             'out': [('f','float'),
                     ('fvec','float',214),
                     # ('Jacobian','float',(214,n)),
                     ],
             'params': {'combine_component_func': sum_squares,
                        }, 
             }

out = [('x','float',n),
      ('x_on_cube','float',n),
      ('pt_id','int'),
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
      ]

gen_specs = {'f': aposmm_logic,
             'in': [o[0] for o in out] + ['fvec', 'f', 'returned'],
             # 'in': [o[0] for o in out] + ['f', 'returned'],
             'out': out,
             'params': {'lb': -2*np.ones(3),
                        'ub':  2*np.ones(3),
                        'initial_sample': 2,
                        'localopt_method': 'LN_BOBYQA',
                        'rk_const': ((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
                        'xtol_rel': 1e-3,
                        'min_batch_size': len(allocation_specs['worker_ranks']),
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
    np.save('H_after_100_evals',H)
