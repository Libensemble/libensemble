# """
# Runs libEnsemble with uniform random sampling and writes results into sim dirs.
#   tests  per-calculation sim_dir capabilities
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_worker_exceptions.py
#    python3 test_worker_exceptions.py --nworkers 3 --comms local
#    python3 test_worker_exceptions.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np
import os

from libensemble.libE import libE
from libensemble.tests.regression_tests.support import write_sim_func as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.manager import ManagerException

nworkers, is_manager, libE_specs, _ = parse_args()

sim_input_dir = './sim_input_dir'
dir_to_copy = sim_input_dir + '/copy_this'
dir_to_symlink = sim_input_dir + '/symlink_this'
e_ensemble = './ensemble_calcdirs_w' + str(nworkers) + '_' + libE_specs.get('comms')
print('attempting to use ensemble dir: ', e_ensemble, flush=True)
print('previous dir contains ', len(os.listdir(e_ensemble)), ' items.', flush=True)

assert os.path.isdir(e_ensemble), \
    "Previous ensemble directory doesn't exist. Can't test exception."
assert len(os.listdir(e_ensemble)), \
    "Previous ensemble directory doesn't have any contents. Can't catch exception."

libE_specs['sim_dirs_make'] = True
libE_specs['ensemble_dir_path'] = e_ensemble
libE_specs['use_worker_dirs'] = False
libE_specs['sim_dir_copy_files'] = [dir_to_copy]
libE_specs['sim_dir_symlink_files'] = [dir_to_symlink]

libE_specs['abort_on_exception'] = False

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'out': [('x', float, (1,))],
             'user': {'gen_batch_size': 20,
                      'lb': np.array([-3]),
                      'ub': np.array([3]),
                      }
             }

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': 21}

return_flag = 1
try:
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                persis_info, libE_specs=libE_specs)
except ManagerException as e:
    print("Caught deliberate exception: {}".format(e))
    return_flag = 0

if is_manager:
    assert return_flag == 0
