from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.message_numbers import (UNSET_TAG, WORKER_KILL_ON_ERR,
                                         MAN_SIGNAL_FINISH, WORKER_DONE,
                                         TASK_FAILED, WORKER_KILL_ON_TIMEOUT)
import numpy as np

__all__ = ['executor_hworld']

# Alt send values through X
sim_count = 0


def polling_loop(comm, exctr, task, timeout_sec=3.0, delay=0.3):
    import time

    calc_status = UNSET_TAG  # Sim func determines status of libensemble calc - returned to worker

    while task.runtime < timeout_sec:
        time.sleep(delay)

        # print('Probing manager at time: ', task.runtime)
        exctr.manager_poll(comm)
        if exctr.manager_signal == 'finish':
            exctr.kill(task)
            calc_status = MAN_SIGNAL_FINISH  # Worker will pick this up and close down
            print('Task {} killed by manager on worker {}'.format(task.id, exctr.workerID))
            break

        # print('Polling task at time', task.runtime)
        task.poll()
        if task.finished:
            break
        elif task.state == 'RUNNING':
            print('Task {} still running on worker {} ....'.format(task.id, exctr.workerID))

        # Check output file for error
        # print('Checking output file for error at time:', task.runtime)
        if task.stdout_exists():
            if 'Error' in task.read_stdout():
                print("Found (deliberate) Error in ouput file - cancelling "
                      "task {} on worker {}".format(task.id, exctr.workerID))
                exctr.kill(task)
                calc_status = WORKER_KILL_ON_ERR
                break

    # After exiting loop
    if task.finished:
        print('Task {} done on worker {}'.format(task.id, exctr.workerID))
        # Fill in calc_status if not already
        if calc_status == UNSET_TAG:
            if task.state == 'FINISHED':  # Means finished succesfully
                calc_status = WORKER_DONE
            elif task.state == 'FAILED':
                calc_status = TASK_FAILED
            # elif task.state == 'USER_KILLED':
            #     calc_status = WORKER_KILL
    else:
        # assert task.state == 'RUNNING', "task.state expected to be RUNNING. Returned: " + str(task.state)
        print("Task {} timed out - killing on worker {}".format(task.id, exctr.workerID))
        exctr.kill(task)
        if task.finished:
            print('Task {} done on worker {}'.format(task.id, exctr.workerID))
        calc_status = WORKER_KILL_ON_TIMEOUT

    return task, calc_status


def distrib_ml_eval_model(H, persis_info, gen_specs, libE_info):
    exctr = MPIExecutor.executor

    task = exctr.submit(calc_type='gen', num_procs=cores, hyperthreads=True)

    if wait:
        task.wait()
        if not task.finished:
            calc_status = UNSET_TAG
        if task.state == 'FINISHED':
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            calc_status = TASK_FAILED

    else:
        task, calc_status = polling_loop(comm, exctr, task, timeout)

    # This is just for testing at calling script level - status of each task
    H_o['cstat'] = calc_status

    return H_o, persis_info, calc_status
