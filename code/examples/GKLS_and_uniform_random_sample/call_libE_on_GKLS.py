# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. (You will need to run "make gkls_single" in libensemble/examples/GKLS_sim_src/
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

# Declare the objective
sys.path.append('./GKLS_sim_src')
from GKLS_obj import call_GKLS_with_random_pause as obj_func


# Below is the generating function that is called by LibEnsemble to generate
# points to be evaluated. In this case, it is just a uniform random sample
# over params['lb'] to params['ub']
def uniform_random_sample(g_in,gen_out,params):
    ub = params['ub']
    lb = params['lb']

    n = len(lb)
    b = params['gen_batch_size']

    O = np.zeros(b, dtype=gen_out)
    for i in range(0,b):
        x = np.random.uniform(lb,ub,(1,n))

        O['x'][i] = x
        O['priority'][i] = np.random.uniform(0,1)

    return O

# Below is a function that will be passed to the workers to tell them how
# to combine multiple residuals in order to get a scalar objective value
def combine_fvec(F):
    return(np.sum(F))


### Declare the run parameters/functions
max_sim_evals = 100


#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': [obj_func],
             'in': ['x'],
             'out': [('fvec',float,3),
                     ('f',float),
                    ],
             'params': {'number_of_minima': 10,
                        'problem_dimension': 2,
                        'problem_number': 1,
                        'combine_component_func': combine_fvec,
                        'uniform_random_pause_ub': 0.01,
                        'sim_dir': './GKLS_sim_src'}, # to be copied by each worker 
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': [],
             'out': [('x',float,2),
                     ('priority',float),
                    ],
             'params': {'lb': np.array([0,0]),
                        'ub': np.array([1,1]),
                        'gen_batch_size': max_sim_evals,
                       },
             'num_inst': 1,
             'batch_mode': True,
             }

# Tell LibEnsemble when to stop
exit_criteria = {'sim_eval_max': max_sim_evals, # must be provided
                 'elapsed_wallclock_time': 100,
                 'stop_val': ('f', -1), # key must be in sim_specs['out'] or gen_specs['out'] 
                }

np.random.seed(1)
# Perform the run
H = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    filename = 'GKLS_results_after_evals=' + str(max_sim_evals) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)
