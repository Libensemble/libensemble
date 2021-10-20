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
# from libensemble.calc_info import CalcInfo
# from libensemble.executors.executor import Executor
# from libensemble.resources.resources import Resources
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, TASK_FAILED
from libensemble.libE import libE
from libensemble.sim_funcs.executor_hworld import executor_hworld as sim_f
import libensemble.sim_funcs.six_hump_camel as six_hump_camel
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.tests.regression_tests.common import build_simfunc

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 3 4

nworkers, is_manager, libE_specs, _ = parse_args()

libE_specs['disable_resource_manager'] = True

USE_BALSAM = False

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

if USE_BALSAM:
    from libensemble.executors.balsam_executor import BalsamMPIExecutor

    exctr = BalsamMPIExecutor()
else:
    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor()
exctr.register_app(full_path=sim_app, calc_type='sim')  # Default 'sim' app - backward compatible
exctr.register_app(full_path=sim_app2, app_name='six_hump_camel')  # Named app

# if nworkers == 3:
#    CalcInfo.keep_worker_stat_files = True # Testing this functionality
# else:
#    CalcInfo.keep_worker_stat_files = False # Testing this functionality

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

exit_criteria = {'elapsed_wallclock_time': 30}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_manager:
    print('\nChecking expected task status against Workers ...\n')

    # Expected list: Last is zero as will not be entered into H array on
    # manager kill - but should show in the summary file.
    # Repeat expected lists nworkers times and compare with list of status's
    # received from workers
    calc_status_list_in = np.asarray(
        [WORKER_DONE, WORKER_KILL_ON_ERR, WORKER_DONE, WORKER_KILL_ON_TIMEOUT, TASK_FAILED, 0]
    )
    calc_status_list = np.repeat(calc_status_list_in, nworkers)

    # For debug
    print("Expecting: {}".format(calc_status_list))
    print("Received:  {}\n".format(H['cstat']))

    assert np.array_equal(H['cstat'], calc_status_list), "Error - unexpected calc status. Received: " + str(H['cstat'])

    # Check summary file:
    print('Checking expected task status against task summary file ...\n')

    calc_desc_list_in = [
        'Completed',
        'Worker killed task on Error',
        'Completed',
        'Worker killed task on Timeout',
        'Task Failed',
        'Manager killed on finish',
    ]

    # Repeat N times for N workers and insert Completed at start for generator
    calc_desc_list = ['Completed'] + calc_desc_list_in * nworkers
    # script_name = os.path.splitext(os.path.basename(__file__))[0]
    # short_name = script_name.split("test_", 1).pop()
    # summary_file_name = short_name + '.libe_summary.txt'
    # with open(summary_file_name,'r') as f:
    #     i=0
    #     for line in f:
    #         if "Status:" in line:
    #             _, file_status = line.partition("Status:")[::2]
    #             print("Expected: {}   Filestatus: {}".format(calc_desc_list[i], file_status.strip()))
    #             assert calc_desc_list[i] == file_status.strip(), "Status does not match file"
    #             i+=1

    print("\n\n\nRun completed.")
