from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.message_numbers import (UNSET_TAG, WORKER_KILL_ON_ERR,
                                         MAN_SIGNAL_FINISH, WORKER_DONE,
                                         TASK_FAILED, WORKER_KILL_ON_TIMEOUT)
import numpy as np
import os

__all__ = ['executor_hworld']

# Alt send values through X
returned_count = 0


def custom_polling_loop(exctr, task, timeout_sec=5.0, delay=0.3):
    import time

    calc_status = UNSET_TAG  # Sim func determines status of libensemble calc - returned to worker

    while task.runtime < timeout_sec:
        time.sleep(delay)

        # print('Probing manager at time: ', task.runtime)
        exctr.manager_poll()
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
                print("Found (deliberate) Error in output file - cancelling "
                      "task {} on worker {}".format(task.id, exctr.workerID))
                exctr.kill(task)
                calc_status = WORKER_KILL_ON_ERR
                break

    # After exiting loop
    if task.finished:
        print('Task {} done on worker {}'.format(task.id, exctr.workerID))
        # Fill in calc_status if not already
        if calc_status == UNSET_TAG:
            if task.state == 'FINISHED':  # Means finished successfully
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
    use_balsam = 'balsam_test' in sim_specs['user']

    args_for_sim = 'sleep 1'
    # pref send this in X as a sim_in from calling script
    global returned_count
    returned_count += 1
    timeout = 6.0
    wait = False
    launch_shc = False
    if returned_count == 1:
        args_for_sim = 'sleep 1'  # Should finish
    elif returned_count == 2:
        args_for_sim = 'sleep 1 Error'  # Worker kill on error
    if returned_count == 3:
        wait = True
        args_for_sim = 'sleep 1'  # Should finish
        launch_shc = True
    elif returned_count == 4:
        args_for_sim = 'sleep 8'  # Worker kill on timeout
        timeout = 1.0
    elif returned_count == 5:
        args_for_sim = 'sleep 2 Fail'  # Manager kill - if signal received else completes
    elif returned_count == 6:
        args_for_sim = 'sleep 60'  # Manager kill - if signal received else completes
        timeout = 65.0

    if use_balsam:
        task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim,
                            hyperthreads=True, machinefile='notused', stdout='notused',
                            wait_on_start=True)
    else:
        task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim, hyperthreads=True)

    if wait:
        task.wait()
        if not task.finished:
            calc_status = UNSET_TAG
        if task.state == 'FINISHED':
            calc_status = WORKER_DONE
        elif task.state == 'FAILED':
            calc_status = TASK_FAILED

    else:
        if returned_count >= 2 and not use_balsam:
            calc_status = exctr.polling_loop(task, timeout=timeout, delay=0.3, poll_manager=True)
            if returned_count == 2 and task.stdout_exists() and 'Error' in task.read_stdout():
                calc_status = WORKER_KILL_ON_ERR

        else:
            task, calc_status = custom_polling_loop(exctr, task, timeout)

    if use_balsam:
        task.read_file_in_workdir('ensemble.log')
        try:
            task.read_stderr()
        except ValueError:
            pass

        task = exctr.submit(app_name='sim_hump_camel_dry_run', num_procs=cores, app_args=args_for_sim,
                            hyperthreads=True, machinefile='notused', stdout='notused',
                            wait_on_start=True, dry_run=True, stage_inout=os.getcwd())

        task.poll()
        task.wait()

    # This is temp - return something - so doing six_hump_camel_func again...
    batch = len(H['x'])
    H_o = np.zeros(batch, dtype=sim_specs['out'])
    for i, x in enumerate(H['x']):
        H_o['f'][i] = six_hump_camel_func(x)
        if launch_shc:
            # Test launching a named app.
            app_args = ' '.join(str(val) for val in list(x[:]))
            task = exctr.submit(app_name='six_hump_camel', num_procs=1, app_args=app_args)
            task.wait()
            output = np.float64(task.read_stdout())
            assert np.isclose(H_o['f'][i], output)

    # This is just for testing at calling script level - status of each task
    H_o['cstat'] = calc_status

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
