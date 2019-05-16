# """
# A test of libEnsemble exception handling.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_calc_exception.py
#    python3 test_calc_exception.py --nworkers 3 --comms local
#    python3 test_calc_exception.py --nworkers 3 --comms tcp
#
#
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

from libensemble.libE import libE
from libensemble.libE_manager import ManagerException
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.tests.regression_tests.common import parse_args, per_worker_stream

nworkers, is_master, libE_specs, _ = parse_args()


# Define sim_func
def six_hump_camel_err(H, persis_info, sim_specs, _):
    raise Exception('Deliberate error')


sim_specs = {'sim_f': six_hump_camel_err, 'in': ['x'], 'out': [('f', float)]}
gen_specs = {'gen_f': gen_f,
             'in': ['sim_id'],
             'out': [('x', float, 2)],
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2]),
             'gen_batch_size': 10}

persis_info = per_worker_stream({}, nworkers + 1)

exit_criteria = {'elapsed_wallclock_time': 10}

libE_specs['abort_on_exception'] = False

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
