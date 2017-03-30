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


def uniform_random_sample(params):
    ub = params['ub']
    lb = params['lb']
    n = len(lb)

    x = np.random.uniform(0,1,n)*(ub-lb)+lb
    return(x)

def combine_fvec(F):
    return(np.sum(F))


### Declare the run parameters/functions
c = {}
c['comm'] = MPI.COMM_WORLD
c['color'] = 0

allocation_specs = {'manager_ranks': set([0]), 
                    'worker_ranks': set(range(1,c['comm'].Get_size()))
                   }

sim_specs = {'f': [obj_func],
             'in': ['x'],
             'out': {'fvec': 3, 'f': 1},
             'params': {'number_of_minima': 10,
                        'problem_dimension': 2,
                        'problem_number': 2,
                        'combine_component_func': combine_fvec,
                        'obj_dir': './GKLS'}, # to be copied by each worker 
             }

gen_specs = {'f': uniform_random_sample,
             'in': [],
             'out': {'x': 2, 'priorities': 1, 'run_number': 1},
             'params': {'lb': np.array([0,0]),
                        'ub': np.array([1,1])},
             }

failure_processing = {}

exit_criteria = {'sim_eval_max': 10,        
                 'min_sim_f_val': -0.5,    
                 'elapsed_clock_time': 100,
                  }

np.random.seed(1)
# Perform the run
H = libE(c, allocation_specs, sim_specs, gen_specs, failure_processing, exit_criteria)
