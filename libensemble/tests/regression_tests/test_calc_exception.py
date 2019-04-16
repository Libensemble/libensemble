import sys, os             # for adding to path
import numpy as np

from libensemble.libE import libE
from libensemble.libE_manager import ManagerException
from libensemble.tests.regression_tests.common import parse_args

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()
libE_specs['abort_on_exception'] = False


# Define sim_func
def six_hump_camel_err(H, persis_info, sim_specs, _):
    raise Exception('Deliberate error')

from libensemble.tests.regression_tests.support import uniform_random_sample_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import give_each_worker_own_stream 
persis_info = give_each_worker_own_stream({},nworkers+1)

# Import gen_func
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample

# State the objective function and params
sim_specs = {'sim_f': six_hump_camel_err,
             'in': ['x'],
             'out': [('f',float),],
             }

# State the generating function
gen_specs['out'] = [('x',float,2)]
gen_specs['lb'] = np.array([-3,-2])
gen_specs['ub'] = np.array([ 3, 2])
gen_specs['gen_batch_size'] = 10

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
