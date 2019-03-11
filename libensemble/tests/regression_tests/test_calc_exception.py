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

# Import gen_func
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample

script_name = os.path.splitext(os.path.basename(__file__))[0]

# State the objective function and params
sim_specs = {'sim_f': six_hump_camel_err,
             'in': ['x'],
             'out': [('f',float),],
             'save_every_k': 400
             }

# State the generating function
gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             'out': [('x',float,2)],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'num_active_gens': 1,
             'gen_batch_size': 500,
             'save_every_k': 300
             }

# Tell libEnsemble when to stop
exit_criteria = {'gen_max': 501, 'elapsed_wallclock_time': 300}

np.random.seed(1)
persis_info = {}
for i in range(1,nworkers+1):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}

# Perform the run
return_flag = 1
try:
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)
except ManagerException as e:
    print("Caught deliberate exception: {}".format(e))
    return_flag = 0

if is_master:
    if 'abort_on_exception' in libE_specs:
        print("Failed to properly handle error")
    assert return_flag == 0
