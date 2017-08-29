# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html 
# 
# Execute via the following command:
#    mpiexec -np 4 python3 call_6-hump_camel_test_elapsed_time.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np
from math import *

import time

# Import libEnsemble main
sys.path.append('../../src')
from libE import libE

# Import sim_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs'))
from six_hump_camel import six_hump_camel

# Import gen_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from uniform_sampling import uniform_random_sample

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': [six_hump_camel], # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'params': {'constant': 10},
             'save_every_k': 10
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             'out': [('x',float,2),
                    ],
             'params': {'lb': np.array([-3,-2]),
                        'ub': np.array([ 3, 2]),
                        'gen_batch_size': 5,
                       },
             'num_inst': 1,
             'batch_mode': False,
             'save_every_k': 10
             }

# Tell LibEnsemble when to stop
exit_criteria = {'elapsed_wallclock_time': 1}

np.random.seed(1)

# Perform the run
H = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    filename = '6-hump_camel_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)


    # minima = np.array([[ -0.089842,  0.712656],
    #                    [  0.089842, -0.712656],
    #                    [ -1.70361,  0.796084],
    #                    [  1.70361, -0.796084],
    #                    [ -1.6071,   -0.568651],
    #                    [  1.6071,    0.568651]])
    # tol = 0.1
    # for m in minima:
    #     print(np.min(np.sum((H['x']-m)**2,1)))
    #     assert(np.min(np.sum((H['x']-m)**2,1)) < tol)

    #     print("\nLibEnsemble with APOSMM has identified the 6 minima within a tolerance " + str(tol))


