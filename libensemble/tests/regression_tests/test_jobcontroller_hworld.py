import os # for adding to path
import numpy as np

# Import libEnsemble items for this test
# from libensemble.calc_info import CalcInfo
from libensemble.controller import JobController
from libensemble.resources import Resources
from libensemble.message_numbers import *
from libensemble.libE import libE
from libensemble.sim_funcs.job_control_hworld import job_control_hworld as sim_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.tests.regression_tests.common import build_simfunc, parse_args, give_each_worker_own_stream

nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    quit()

USE_BALSAM = False

NCORES = 1
sim_app = './my_simjob.x'
if not os.path.isfile(sim_app):
    build_simfunc()

if USE_BALSAM:
    from libensemble.balsam_controller import BalsamJobController
    jobctrl = BalsamJobController(auto_resources=True)
else:
    from libensemble.mpi_controller import MPIJobController
    jobctrl = MPIJobController(auto_resources=False)
jobctrl.register_calc(full_path=sim_app, calc_type='sim')

# if nworkers == 3:
#    CalcInfo.keep_worker_stat_files = True # Testing this functionality
# else:
#    CalcInfo.keep_worker_stat_files = False # Testing this functionality

sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    'out': [
        ('f', float),
        ('cstat', int),],
    'save_every_k': 400,
    'cores': NCORES,}

gen_specs = {
    'gen_f': gen_f,
    'in': ['sim_id'],
    'out': [('x', float, (2,))],
    'lb': np.array([-3, -2]),
    'ub': np.array([3, 2]),
    'gen_batch_size': 5*nworkers,
    'batch_mode': True,
    'num_active_gens': 1,
    'save_every_k': 20,}

persis_info = give_each_worker_own_stream({}, nworkers+1)

exit_criteria = {'elapsed_wallclock_time': 15}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:
    print('\nChecking expected job status against Workers ...\n')

    #Expected list: Last is zero as will not be entered into H array on manager kill - but should show in the summary file.
    #Repeat expected lists nworkers times and compare with list of status's received from workers
    calc_status_list_in = np.asarray([
        WORKER_DONE, WORKER_KILL_ON_ERR, WORKER_KILL_ON_TIMEOUT, JOB_FAILED,
        0])
    calc_status_list = np.repeat(calc_status_list_in, nworkers)

    #For debug
    print("Expecting: {}".format(calc_status_list))
    print("Received:  {}\n".format(H['cstat']))

    assert np.array_equal(
        H['cstat'],
        calc_status_list), "Error - unexpected calc status. Received: "+str(
            H['cstat'])

    #Check summary file:
    print('Checking expected job status against job summary file ...\n')

    calc_desc_list_in = [
        'Completed', 'Worker killed job on Error',
        'Worker killed job on Timeout', 'Job Failed',
        'Manager killed on finish']
    #Repeat N times for N workers and insert Completed at start for generator
    calc_desc_list = ['Completed']+calc_desc_list_in*nworkers
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
