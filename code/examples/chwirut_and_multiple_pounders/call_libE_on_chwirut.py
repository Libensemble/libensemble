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
import numpy as np

sys.path.append('../../src')
from libE import libE

sys.path.append('./GKLS_sim_src')
from GKLS_obj import call_GKLS as obj_func

from aposmm_logic import aposmm_logic

### Declare the run parameters/functions
c = {}
c['comm'] = MPI.COMM_WORLD
c['color'] = 0

allocation_specs = {'manager_ranks': set([0]), 
                    'worker_ranks': set(range(1,c['comm'].Get_size()))
                   }

sim_specs = {'f': [obj_func],
             'in': ['x'],
             'out': [('fvec','float',214),
                     ('f','float'),
                     ('Jacobian','float',(3,214)),
                     ],
             'params': {'combine_component_func': np.linalg.norm,
                        'obj_dir': './dir'}, # to be copied by each worker 
             }

gen_specs = {'f': aposmm_logic,
             'in': ['x', 'f', 'local_pt', 'run_number', 'dist_to_unit_bounds',
                    'dist_to_better_l', 'dist_to_better_s', 'ind_of_better_l',
                    'ind_of_better_s', 'started_run', 'active', 'local_min', ],
             'out': [('x','float',3),
                     ('priority','float'),
                     ('run_number','int'),
                     ('local_pt','bool')
                     ('dist_to_unit_bounds','float'),
                     ('dist_to_better_l','float'),
                     ('dist_to_better_s','float'),
                     ('ind_of_better_l','int'),
                     ('ind_of_better_s','int'),
                     ('started_run','bool'),
                     ('active','bool')
                     ('local_min','bool')
                     ],
             'params': {'lb': np.array([0,0]),
                        'ub': np.array([1,1]),
                        'initial_sample': 10,
                        },
             }

failure_processing = {}

exit_criteria = {'sim_eval_max': 20, # must be provided
                  }

np.random.seed(1)
# Perform the run
H = libE(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria)
