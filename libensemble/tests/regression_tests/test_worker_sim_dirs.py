# """
# Runs libEnsemble on a gen_f that is missing necessary information; tests libE worker exception raising
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
import shutil

from libensemble.libE import libE
from libensemble.tests.regression_tests.support import write_func as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.utils import parse_args, add_unique_random_streams

nworkers, is_master, libE_specs, _ = parse_args()

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'out': [('x', float, (1,))],
             'user': {'gen_batch_size': 20,
                      'lb': np.array([-3]),
                      'ub': np.array([3]),
                      }
             }

persis_info = add_unique_random_streams({}, nworkers + 1)

sim_dir = './test_sim_dir'
if not os.path.isdir(sim_dir):
    os.mkdir(sim_dir)

libE_specs['sim_dir'] = sim_dir
libE_specs['do_worker_dir'] = True

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 21}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, libE_specs=libE_specs)

if is_master:
    assert sum(['test_sim_dir_worker' in i for i in os.listdir()]) == nworkers, \
        'Number of worker directories does not match number of workers'

    for i in os.listdir():
        if 'test_sim_dir_worker' in i:
            shutil.rmtree(i)
