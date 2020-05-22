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


def executor_hworld(H, persis_info, sim_specs, libE_info):
    """ Tests launching and polling task and exiting on task finish"""
    exctr = MPIExecutor.executor
    cores = sim_specs['user']['cores']
    comm = libE_info['comm']

    args_for_sim = 'sleep 1'
    # pref send this in X as a sim_in from calling script
    global sim_count
    sim_count += 1
    timeout = 6.0
    if sim_count == 1:
        args_for_sim = 'sleep 1'  # Should finish
    elif sim_count == 2:
        args_for_sim = 'sleep 1 Error'  # Worker kill on error
    elif sim_count == 3:
        args_for_sim = 'sleep 3'  # Worker kill on timeout
        timeout = 1.0
    elif sim_count == 4:
        args_for_sim = 'sleep 1 Fail'  # Manager kill - if signal received else completes
    elif sim_count == 5:
        args_for_sim = 'sleep 18'  # Manager kill - if signal received else completes
        timeout = 20.0

    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim, hyperthreads=True)
    task, calc_status = polling_loop(comm, exctr, task, timeout)

    # assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    # assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)

    # This is temp - return something - so doing six_hump_camel_func again...
    batch = len(H['x'])
    H_o = np.zeros(batch, dtype=sim_specs['out'])
    for i, x in enumerate(H['x']):
        H_o['f'][i] = six_hump_camel_func(x)

    # This is just for testing at calling script level - status of each task
    H_o['cstat'] = calc_status

    # v = np.random.uniform(0, 10)
    # print('About to sleep for :' + str(v))
    # time.sleep(v)

    return H_o, persis_info, calc_status


def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
    term2 = x1*x2
    term3 = (-4+4*x2**2) * x2**2

    return term1 + term2 + term3
