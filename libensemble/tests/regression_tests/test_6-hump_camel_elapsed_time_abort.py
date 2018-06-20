# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html 
# 
# Execute via the following command:
#    mpiexec -np 4 python3 test_6-hump_camel_elapsed_time_abort.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
#sys.path.append('../../src')
from libensemble.libE import libE

# Import sim_func 
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs'))
from libensemble.sim_funcs.six_hump_camel import six_hump_camel

# Import gen_func 
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

script_name = os.path.splitext(os.path.basename(__file__))[0]

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'pause_time': 5,
             # 'save_every_k': 10
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             'out': [('x',float,2),
                    ],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': 5,
             'num_inst': 1,
             'batch_mode': False,
             # 'save_every_k': 10
             }

# Tell libEnsemble when to stop
exit_criteria = {'elapsed_wallclock_time': 3}

np.random.seed(1)

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    eprint(flag)
    eprint(H)
    assert flag == 2
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    # if flag == 2:
    #     print("\n\n\nKilling COMM_WORLD")
    #     MPI.COMM_WORLD.Abort()
