import os
import time
import numpy as np

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

def run_warpX(H, persis_info, sim_specs, libE_info):
    # Setting up variables needed for input and output
    # keys              = variable names
    # x                 = variable values
    # output            = what will be returned to libE

    calc_status = 0  # Returns to worker

    x = H['x']

    # # Get from dictionary if key exists, else return default (e.g. 0)
    # cores = sim_specs['user'].get('cores', None)
    # particle_variance = sim_specs['user'].get('particle_variance', 0)

    # # This is to give a random variance of work-load
    # seed = 1
    # sim_particles = perturb(sim_particles, seed, particle_variance)
    # print('seed: {}   particles: {}'.format(seed, sim_particles))

    # # # At this point you will be in the sim directory (really worker dir) for this worker (eg. sim_1).
    # # # The simdir below is created for each task for this worker.
    # # # Any input needs to be copied into this directory. Currently there is none.
    # # exctr = Executor.executor  # Get Executor

    # # args = str(int(sim_particles)) + ' ' + str(sim_timesteps) + ' ' + str(seed) + ' ' + str(kill_rate)
    # # # task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args, stdout='out.txt', stderr='err.txt')
    # # if cores:
    # #     task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args,
    # #                         stdout='out.txt', stderr='err.txt', wait_on_run=True)
    # # else:
    # #     task = exctr.submit(calc_type='sim', app_args=args, stdout='out.txt',
    # #                         stderr='err.txt', wait_on_run=True)  # Auto-partition

    # if task.finished:
    #     if task.state == 'FINISHED':
    #         print("Task {} completed".format(task.name))
    #         calc_status = WORKER_DONE
    #         if read_last_line(filepath) == "kill":
    #             # Generally mark as complete if want results (completed after poll - before readline)
    #             print("Warning: Task completed although marked as a bad run (kill flag set in forces.stat)")
    #     elif task.state == 'FAILED':
    #         print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
    #         calc_status = TASK_FAILED
    #     elif task.state == 'USER_KILLED':
    #         print("Warning: Task {} has been killed".format(task.name))
    #         calc_status = WORKER_KILL
    #     else:
    #         print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))

    # # os.chdir('../')

    # time.sleep(0.2)
    # try:
    #     data = np.loadtxt(filepath)
    #     # task.read_file_in_workdir(statfile)
    #     final_energy = data[-1]
    # except Exception:
    #     final_energy = np.nan
    #     # print('Warning - Energy Nan')

    outspecs = sim_specs['out']
    output = np.zeros(1, dtype=outspecs)
    output['f'][0] = np.linalg.norm(x) 

    return output, persis_info, calc_status
