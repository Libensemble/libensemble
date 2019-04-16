# """
# Runs libEnsemble on a gen_f that is missing necessary information; tests libE worker exception raising
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_worker_exceptions.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """
import sys, os             # for adding to path
import numpy as np

from libensemble.libE import libE
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.libE_manager import ManagerException
from libensemble.tests.regression_tests.common import parse_args
nworkers, is_master, libE_specs, _ = parse_args()
libE_specs['abort_on_exception'] = False

n = 2

def nan_func(calc_in,persis_info,sim_specs,libE_info):
    H = np.zeros(1,dtype=sim_specs['out'])
    H['f_i'] = np.nan
    H['f'] = np.nan
    return (H, persis_info)

from libensemble.tests.regression_tests.support import give_each_worker_own_stream 
persis_info = give_each_worker_own_stream({},nworkers+1)

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
exit_criteria = {'elapsed_wallclock_time': 10}


# Perform the run
return_flag = 1
try:
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)
except ManagerException as e:
    print("Caught deliberate exception: {}".format(e))
    return_flag = 0

if is_master:
    assert return_flag == 0
