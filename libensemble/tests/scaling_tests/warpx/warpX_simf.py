import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED
from MaxenceLocalIMac import machine_specs
from read_sim_output import read_sim_output

def run_warpX(H, persis_info, sim_specs, libE_info):

    # Setting up variables needed for input and output
    # keys              = variable names
    # x                 = variable values
    # libE_output       = what will be returned to libE

    calc_status = 0  # Returns to worker

    x = H['x'] # Input

    nodes = sim_specs['user'].get('nodes', 1)
    ranks_per_node = sim_specs['user'].get('ranks_per_node', 1)
    input_file = sim_specs['user']['input_filename']
    time_limit = sim_specs['user']['sim_kill_minutes'] * 60.0
    
    exctr = Executor.executor  # Get Executor

    app_args = input_file + ' beam.q_tot=' + str(x[0][0])
    print(app_args)
    os.environ["OMP_NUM_THREADS"] = machine_specs['OMP_NUM_THREADS']

    # testing use of extra_args
    if machine_specs['name'] == 'summit':
        task = exctr.submit(calc_type='sim', extra_args=machine_specs['extra_args'], app_args=app_args,
                            stdout='out.txt', stderr='err.txt', wait_on_run=True)
    else:
        task = exctr.submit(calc_type='sim', num_procs=machine_specs['cores'], app_args=app_args,
                            stdout='out.txt', stderr='err.txt', wait_on_run=True)

    poll_interval = 1  # secs
    while(not task.finished):
        time.sleep(poll_interval)
        task.poll()
        if task.runtime > time_limit:
            task.kill()  # Timeout

    # Set calc_status with optional prints.
    if task.finished:
        if task.state == 'FINISHED':
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
            calc_status = TASK_FAILED
        elif task.state == 'USER_KILLED':
            print("Warning: Task {} has been killed".format(task.name))
        else:
            print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))

    # Extract and calculate what you need to send back
    datafile = 'diags/plotfiles/plt01830/'
    filepath = os.path.join(task.workdir, datafile)
    filepath = os.path.join(task.workdir, datafile)
    time.sleep(0.2)

    try:
        warpX_out = read_sim_output( task.workdir )
    except Exception:
        warpX_out = np.nan
        print('Warning - output is Nan')

    libE_output = np.zeros(1, dtype=sim_specs['out'])

    libE_output['f'] = warpX_out[0]
    #libE_output['energy_std'] = warpX_out[0]
    #libE_output['energy_avg'] = warpX_out[1]
    #libE_output['charge'] = warpX_out[2]

    return libE_output, persis_info, calc_status
