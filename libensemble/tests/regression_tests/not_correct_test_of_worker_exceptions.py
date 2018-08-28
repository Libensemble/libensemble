# """
# Runs libEnsemble on a gen_f that is missing necessary information; tests libE worker exception raising
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_worker_exceptions.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
from libensemble.libE import libE

# Import gen_func
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
script_name = os.path.splitext(os.path.basename(__file__))[0]

n = 2

def nan_func(calc_in,persis_info,sim_specs,libE_info):
    H = np.zeros(1,dtype=sim_specs['out'])
    H['f_i'] = np.nan
    H['f'] = np.nan
    return (H, persis_info)

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': nan_func, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float)],
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x',float,2)],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'initial_sample': 100,
             'batch_mode': True,
             'num_active_gens':1,
             }

# Tell libEnsemble when to stop
exit_criteria = {'elapsed_wallclock_time': 0.1}

np.random.seed(1)
persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}


libE_specs = {'abort_on_worker_exc': True}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info=persis_info, libE_specs=libE_specs)
assert len(H)==0



