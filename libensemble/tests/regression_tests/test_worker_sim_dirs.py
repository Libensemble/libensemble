# """
# Runs libEnsemble with uniform random sampling and writes results into sim dirs.
#   tests per-worker or per-calculation sim_input_dir copying capability
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_worker_exceptions.py
#    python3 test_worker_exceptions.py --nworkers 3 --comms local
#    python3 test_worker_exceptions.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import numpy as np
import os
import shutil

from libensemble.libE import libE
from libensemble.tests.regression_tests.support import write_func as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.utils import parse_args, add_unique_random_streams

nworkers, is_master, libE_specs, _ = parse_args()

sim_input_dir = './test_sim_input_dir'
dir_to_copy = './test_sim_input_dir/copy_this'
dir_to_symlink = './test_sim_input_dir/symlink_this'
dir_to_ignore = './test_sim_input_dir/not_this'
w_ensemble = './w_ensemble'
c_ensemble = './c_ensemble'

for dir in [sim_input_dir, dir_to_symlink, dir_to_ignore]:
    if is_master and not os.path.isdir(dir):
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

for dir in [w_ensemble, c_ensemble]:
    if is_master and os.path.isdir(dir):
        shutil.rmtree(dir)

libE_specs['sim_input_dir'] = sim_input_dir
libE_specs['use_worker_dirs'] = True
libE_specs['ensemble_dir'] = w_ensemble
libE_specs['symlink_input_files'] = ['symlink_this']
libE_specs['copy_input_files'] = ['copy_this']

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

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, libE_specs=libE_specs)

if is_master:
    dir_sum = sum(['worker' in i for i in os.listdir(w_ensemble)])
    assert dir_sum == nworkers, \
        'Number of worker directories ({}) does not match number of workers ({}).'\
        .format(dir_sum, nworkers)

    shutil.rmtree(w_ensemble)

# --- Second Round - Test without worker-dirs ---
libE_specs['use_worker_dirs'] = False
libE_specs['ensemble_dir'] = c_ensemble

H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, libE_specs=libE_specs)

if is_master:
    dir_sum = sum(['worker' in i for i in os.listdir(c_ensemble)])
    assert dir_sum == exit_criteria['sim_max'], \
        'Number of sim directories ({}) does not match max number of simulations ({}).'\
        .format(dir_sum, exit_criteria['sim_max'])

    shutil.rmtree(c_ensemble)
