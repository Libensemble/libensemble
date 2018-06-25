# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html 
# 
# Execute via the following command:
#    mpiexec -np 4 python3 {FILENAME}.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
#sys.path.append('../../src')
from libensemble.libE import libE


# Import sim_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs'))
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f

# Import gen_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform as gen_f

# Import alloc_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/alloc_funcs'))
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': sim_f, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), ('grad',float,2) # This is the output from the function being minimized
                    ],
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x',float,2)],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': 20,
             }

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 40}

np.random.seed(1)

alloc_specs = {'out':[], 'alloc_f':alloc_f}

if MPI.COMM_WORLD.Get_size()==2:
    # Can't do a "persistent worker run" if only one worker
    quit() 

# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
