import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED
from Summit import machine_specs
from read_sim_output import read_sim_output
from write_sim_input import write_sim_input

def run_warpX(H, persis_info, sim_specs, libE_info):

    print(sim_specs['user'])

    # Setting up variables needed for input and output
    # keys              = variable names
    # x                 = variable values
    # libE_output       = what will be returned to libE

    calc_status = 0  # Returns to worker

    nodes = sim_specs['user'].get('nodes', 1)
    ranks_per_node = sim_specs['user'].get('ranks_per_node', 1)
    input_file = sim_specs['user']['input_filename']
    time_limit = sim_specs['user']['sim_kill_minutes'] * 60.0

    if not sim_specs['user']['dummy']:

        exctr = Executor.executor  # Get Executor

        write_sim_input(input_file, H['x'])
        # app_args = input_file + ' beam.q_tot=' + str(-x[0][0])
        app_args = input_file
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

        time.sleep(0.2)

        try:
            warpX_out = read_sim_output( task.workdir )
        except Exception:
            warpX_out = np.nan
            print('Warning - output is Nan')

    else:
        # Build a custom function to minimize. This one has two local minima
        xmin = 1.e-13
        xmax = 3.e-12
        xopt = 1.e-12
        warpX_out = np.zeros(len(sim_specs['out']))
        warpX_out[0] = (x[0][0]-.5e-12**2) * (x[0][0]-1.e-12)**2 * (x[0][0]-2.5e-12)**2 * 1.e60

    libE_output = np.zeros(1, dtype=sim_specs['out'])
    libE_output['f'] = warpX_out[0]
    libE_output['energy_std'] = warpX_out[1]
    libE_output['energy_avg'] = warpX_out[2]
    libE_output['charge'] = warpX_out[3]
    libE_output['emittance'] = warpX_out[4]

    return libE_output, persis_info, calc_status
