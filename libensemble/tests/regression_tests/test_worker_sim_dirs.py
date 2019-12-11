# """
# Runs libEnsemble with uniform random sampling and writes results into sim dirs.
#   tests per-worker or per-calculation sim_dir copying capability
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


def cleanup(dir):
    here = os.getcwd()
    for i in os.listdir(dir):
        if 'test_sim_dir' in i:
            os.chdir(dir)
            shutil.rmtree(i)
            os.chdir(here)


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
ensemble_dir = './ensemble'

for dir in [sim_dir, ensemble_dir]:
    if is_master and not os.path.isdir(dir):
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass

if is_master:
    cleanup(ensemble_dir)

libE_specs['sim_dir'] = sim_dir
libE_specs['do_worker_dir'] = True
libE_specs['sim_dir_prefix'] = ensemble_dir
libE_specs['sym_link_to_input'] = True

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 21}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, libE_specs=libE_specs)

if is_master:
    dir_sum = sum(['test_sim_dir_worker' in i for i in os.listdir(ensemble_dir)])
    assert dir_sum == nworkers, \
        'Num worker directories ({}) does not match number of workers ({}).'\
        .format(dir_sum, nworkers)

    cleanup(ensemble_dir)
