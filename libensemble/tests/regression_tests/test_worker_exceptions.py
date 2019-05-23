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

from libensemble.libE import libE
from libensemble.tests.regression_tests.support import nan_func as sim_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.libE_manager import ManagerException
from libensemble.tests.regression_tests.common import parse_args, per_worker_stream

nworkers, is_master, libE_specs, _ = parse_args()
n = 2

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, 2)],
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2]),
             'initial_sample': 100,
             'batch_mode': True,
             'num_active_gens': 1}

persis_info = per_worker_stream({}, nworkers + 1)

libE_specs['abort_on_exception'] = False

# Tell libEnsemble when to stop
exit_criteria = {'elapsed_wallclock_time': 10}

# Perform the run
return_flag = 1
try:
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                persis_info, libE_specs=libE_specs)
except ManagerException as e:
    print("Caught deliberate exception: {}".format(e))
    return_flag = 0

if is_master:
    assert return_flag == 0
