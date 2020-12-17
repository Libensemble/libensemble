from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.message_numbers import (
                                         MAN_SIGNAL_FINISH, WORKER_DONE,
                                         TASK_FAILED, WORKER_KILL_ON_TIMEOUT)
import os
import time
import glob
import numpy as np

__all__ = ['distrib_ml_eval_model']

# Alt send values through X

# def polling_loop(comm, exctr, task, timeout_sec=3.0, delay=0.3):
#     import time
#
#     calc_status = UNSET_TAG  # Sim func determines status of libensemble calc - returned to worker
#
#     while task.runtime < timeout_sec:
#         time.sleep(delay)
#
#         # print('Probing manager at time: ', task.runtime)
#         exctr.manager_poll(comm)
#         if exctr.manager_signal == 'finish':
#             exctr.kill(task)
#             calc_status = MAN_SIGNAL_FINISH  # Worker will pick this up and close down
#             print('Task {} killed by manager on worker {}'.format(task.id, exctr.workerID))
#             break
#
#         # print('Polling task at time', task.runtime)
#         task.poll()
#         if task.finished:
#             break
#         elif task.state == 'RUNNING':
#             print('Task {} still running on worker {} ....'.format(task.id, exctr.workerID))
#
#         # Check output file for error
#         # print('Checking output file for error at time:', task.runtime)
#         if task.stdout_exists():
#             if 'Error' in task.read_stdout():
#                 print("Found (deliberate) Error in ouput file - cancelling "
#                       "task {} on worker {}".format(task.id, exctr.workerID))
#                 exctr.kill(task)
#                 calc_status = WORKER_KILL_ON_ERR
#                 break
#
#     # After exiting loop
#     if task.finished:
#         print('Task {} done on worker {}'.format(task.id, exctr.workerID))
#         # Fill in calc_status if not already
#         if calc_status == UNSET_TAG:
#             if task.state == 'FINISHED':  # Means finished succesfully
#                 calc_status = WORKER_DONE
#             elif task.state == 'FAILED':
#                 calc_status = TASK_FAILED
#             # elif task.state == 'USER_KILLED':
#             #     calc_status = WORKER_KILL
#     else:
#         # assert task.state == 'RUNNING', "task.state expected to be RUNNING. Returned: " + str(task.state)
#         print("Task {} timed out - killing on worker {}".format(task.id, exctr.workerID))
#         exctr.kill(task)
#         if task.finished:
#             print('Task {} done on worker {}'.format(task.id, exctr.workerID))
#         calc_status = WORKER_KILL_ON_TIMEOUT
#
#     return task, calc_status


def distrib_ml_eval_model(H, persis_info, gen_specs, libE_info):

    comm = libE_info['comm']
    num_procs = gen_specs['user']['num_procs']
    app_args = gen_specs['user']['app_args']
    dry_run = gen_specs['user']['dry_run']
    time_limit = gen_specs['user']['time_limit']

    exctr = MPIExecutor.executor
    task = exctr.submit(calc_type='gen', num_procs=num_procs, app_args=app_args,
                        stdout='out.txt', stderr='err.txt', hyperthreads=True,
                        dry_run=dry_run)

    poll_interval = 30  # secs
    while(not task.finished):
        if task.runtime > time_limit:
            task.kill()  # Timeout
            calc_status = WORKER_KILL_ON_TIMEOUT
        if exctr.manager_signal == 'finish':
            task.kill()
            calc_status = MAN_SIGNAL_FINISH
        else:
            time.sleep(poll_interval)
            task.poll()
            exctr.manager_poll(comm)

    if task.finished:
        if task.state == 'FINISHED':
            print("Task {} completed".format(task.name))
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
            calc_status = TASK_FAILED
        elif task.state == 'USER_KILLED':
            print("Warning: Task {} has been killed".format(task.name))
            calc_status = WORKER_KILL
        else:
            print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))
            calc_status = UNSET_TAG

    time.sleep(0.2)
    model_file = glob.glob('./final*')
    assert len(model_file), "Keras Application did not write final output to file."

    # task, calc_status = polling_loop(comm, exctr, task, timeout)

    # This is just for testing at calling script level - status of each task
    H_o = np.zeros(1, dtype=sim_specs['out'])
    H_o['cstat'] = calc_status
    H_o['model_file'] = os.path.abspath(model_file[0])

    return H_o, persis_info, calc_status
