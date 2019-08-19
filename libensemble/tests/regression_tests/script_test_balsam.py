# This script is submitted as an app and job to Balsam, and jobs
#   are launched based on this app. This submission is via '
#   balsam launch' executed in the test_balsam.py script.

import os
import numpy as np
import multiprocessing
import mpi4py
from mpi4py import MPI

from libensemble.balsam_controller import BalsamJobController
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, JOB_FAILED
from libensemble.libE import libE
from libensemble.sim_funcs.job_control_hworld_balsam import job_control_hworld as sim_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.tests.regression_tests.common import per_worker_stream

mpi4py.rc.recv_mprobe = False  # Disable matching probes


# Slighty different due to not executing in /regression_tests
def build_simfunc():
    import subprocess
    print('Balsam job launched in: {}'.format(os.getcwd()))
    buildstring = 'mpicc -o my_simjob.x libensemble/tests/unit_tests/simdir/my_simjob.c'
    subprocess.check_call(buildstring.split())


libE_specs = {'comm': MPI.COMM_WORLD, 'color': 0, 'comms': 'mpi'}

nworkers = MPI.COMM_WORLD.Get_size() - 1
is_master = MPI.COMM_WORLD.Get_rank() == 0

cores_per_job = 1
logical_cores = multiprocessing.cpu_count()
cores_all_jobs = nworkers*cores_per_job

if is_master:
    print('\nCores req: {} Cores avail: {}\n'.format(cores_all_jobs,
                                                     logical_cores))

sim_app = './my_simjob.x'
if not os.path.isfile(sim_app):
    build_simfunc()

jobctrl = BalsamJobController(auto_resources=False)
jobctrl.register_calc(full_path=sim_app, calc_type='sim')

sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float), ('cstat', int)],
             'save_every_k': 400,
             'cores': cores_per_job}

gen_specs = {'gen_f': gen_f,
             'in': ['sim_id'],
             'out': [('x', float, (2,))],
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2]),
             'gen_batch_size': nworkers,
             'batch_mode': True,
             'num_active_gens': 1,
             'save_every_k': 20}

persis_info = per_worker_stream({}, nworkers + 1)

exit_criteria = {'elapsed_wallclock_time': 30}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                            persis_info, libE_specs=libE_specs)

if is_master:
    print('\nChecking expected job status against Workers ...\n')

    # Expected list: Last is zero as will not be entered into H array on
    # manager kill - but should show in the summary file.
    # Repeat expected lists nworkers times and compare with list of status's
    # received from workers
    calc_status_list_in = np.asarray([WORKER_DONE, WORKER_KILL_ON_ERR,
                                      WORKER_KILL_ON_TIMEOUT,
                                      JOB_FAILED, 0])
    calc_status_list = np.repeat(calc_status_list_in, nworkers)

    # For debug
    print("Expecting: {}".format(calc_status_list))
    print("Received:  {}\n".format(H['cstat']))

    assert np.array_equal(H['cstat'], calc_status_list), "Error - unexpected calc status. Received: " + str(H['cstat'])

    # Check summary file:
    print('Checking expected job status against job summary file ...\n')

    calc_desc_list_in = ['Completed', 'Worker killed job on Error',
                         'Worker killed job on Timeout', 'Job Failed',
                         'Manager killed on finish']

    # Repeat N times for N workers and insert Completed at start for generator
    calc_desc_list = ['Completed'] + calc_desc_list_in*nworkers

    print("\n\n\nRun completed.")
