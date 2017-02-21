from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
import numpy as np

sys.path.append('../src')
from libE import libE

sys.path.append('../libensemble_applications/GKLS')
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

history = []

allocation_specs = {'manager_ranks': set([0]), 
        'lead_worker_ranks': set(range(1,c['comm'].Get_size()))}

sim_f_params = {'n': 2, 
        'm': 1, 
        'combine_func': combine_fvec,
        'lb': np.array([0,0]),
        'ub': np.array([1,1]),
        'obj_dir': '../libensemble_applications/GKLS', # to be copied by each lead worker
        'sim_data': {'number_of_minima': 10,
                     'problem_dimension': 2,
                     'problem_number': 2}}

sim_specs = {'sim_f': obj_func,
        'sim_f_params': sim_f_params,
        'gen_f': uniform_random_sample,
        'gen_f_params': {'lb': np.array([0,0]),
                         'ub': np.array([1,1])}
        }

failure_processing = {}

exit_criteria = {'sim_eval_max': 1000, # Stop after this many sim evaluations
        'min_sim_f_val': -0.5,       # Stop when sim value is less than this 
        'elapsed_clock_time': 100,   # Don't start new sim evals after this much elapsed time (since first point given to worker)
        }



np.random.seed(1)
# Perform the run
H = libE(c, history, allocation_specs, sim_specs, failure_processing, exit_criteria)
