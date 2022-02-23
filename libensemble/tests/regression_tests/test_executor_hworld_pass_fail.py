"""
Runs libEnsemble testing the executor functionality.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_executor_hworld.py
   python3 test_executor_hworld.py --nworkers 3 --comms local
   python3 test_executor_hworld.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import os
import numpy as np
import multiprocessing

# Import libEnsemble items for this test
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, TASK_FAILED
from libensemble.libE import libE
from libensemble.sim_funcs.executor_hworld import executor_hworld as sim_f
import libensemble.sim_funcs.six_hump_camel as six_hump_camel
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.tests.regression_tests.common import build_simfunc
from libensemble.executors.mpi_executor import MPIExecutor


# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_OS_SKIP: OSX
# TESTSUITE_NPROCS: 2 3 4
# TESTSUITE_OMPI_SKIP: true

nworkers, is_manager, libE_specs, _ = parse_args()

libE_specs['disable_resource_manager'] = True

cores_per_task = 1
logical_cores = multiprocessing.cpu_count()
cores_all_tasks = nworkers * cores_per_task

if cores_all_tasks > logical_cores:
    disable_resource_manager = True
    mess_resources = 'Oversubscribing - Resource manager disabled'
elif libE_specs.get('comms', False) == 'tcp':
    disable_resource_manager = True
    mess_resources = 'TCP comms does not support resource management. Resource manager disabled'
else:
    disable_resource_manager = False
    mess_resources = 'Resource manager enabled'

if is_manager:
    print('\nCores req: {} Cores avail: {}\n  {}\n'.format(cores_all_tasks, logical_cores, mess_resources))

sim_app = './my_simtask.x'
if not os.path.isfile(sim_app):
    build_simfunc()
sim_app2 = six_hump_camel.__file__

exctr = MPIExecutor()

exctr.register_app(full_path=sim_app, calc_type='sim')  # Default 'sim' app - backward compatible
exctr.register_app(full_path=sim_app2, app_name='six_hump_camel')  # Named app

sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    'out': [('f', float), ('cstat', int)],
    'user': {'cores': cores_per_task},
}

gen_specs = {
    'gen_f': gen_f,
    'in': ['sim_id'],
    'out': [('x', float, (2,))],
    'user': {
        'lb': np.array([-3, -2]),
        'ub': np.array([3, 2]),
        'gen_batch_size': nworkers,
    },
}

persis_info = add_unique_random_streams({}, nworkers + 1)

# num returned_count conditions in executor_hworld
exit_criteria = {'sim_max': nworkers * 5}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_manager:
    print('\nChecking expected task status against Workers ...\n')

    calc_status_list_in = np.asarray(
        [WORKER_DONE, WORKER_KILL_ON_ERR, WORKER_DONE, WORKER_KILL_ON_TIMEOUT, TASK_FAILED]
    )
    calc_status_list = np.repeat(calc_status_list_in, nworkers)

    # For debug
    print("Expecting: {}".format(calc_status_list))
    print("Received:  {}\n".format(H['cstat']))

    assert np.array_equal(H['cstat'], calc_status_list), "Error - unexpected calc status. Received: " + str(H['cstat'])

    print("\n\n\nRun completed.")
