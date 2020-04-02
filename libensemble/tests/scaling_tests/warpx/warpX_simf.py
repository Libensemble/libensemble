import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, TASK_FAILED


def run_warpX(H, persis_info, sim_specs, libE_info):

    # Setting up variables needed for input and output
    # keys              = variable names
    # x                 = variable values
    # output            = what will be returned to libE

    calc_status = 0  # Returns to worker

    x = H['x']       # Input

    # nodes = sim_specs['user'].get('nodes', 1)
    # ranks_per_node = sim_specs['user'].get('ranks_per_node', 6)
    input_file = sim_specs['user']['input_filename']
    time_limit = sim_specs['user']['sim_kill_minutes'] * 60.0

    exctr = Executor.executor  # Get Executor

    # task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args,
    #                     stdout='out.txt', stderr='err.txt', wait_on_run=True)

    os.environ["OMP_NUM_THREADS"] = "1"

    # testing use of extra_args
    jsrun_args = '-n 1 -a 2 -g 2 -c 2 --bind=packed:1 --smpiargs="-gpu"'

    task = exctr.submit(calc_type='sim', extra_args=jsrun_args, app_args=input_file,
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
    # datafile = 'forces.stat'
    # filepath = os.path.join(task.workdir, datafile)
    # time.sleep(0.2)
    # try:
        # data = np.loadtxt(filepath)
        # outx = data[-1]
    # except Exception:
        # outx = np.nan
        # print('Warning - output is Nan')

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)

    # output['f'][0] = np.linalg.norm(outx)
    output['f'][0] = np.linalg.norm(x)

    return output, persis_info, calc_status
