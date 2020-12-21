from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.message_numbers import ( UNSET_TAG, MAN_SIGNAL_FINISH, MAN_SIGNAL_KILL,
                                         WORKER_DONE, TASK_FAILED, WORKER_KILL_ON_TIMEOUT)
import os
import sys
import time
import glob
import pickle
import numpy as np

__all__ = ['distrib_ml_build_model']

def distrib_ml_build_model(H, persis_info, gen_specs, libE_info):

    comm = libE_info['comm']
    num_procs = gen_specs['user']['num_procs']
    app_args = gen_specs['user']['app_args']
    dry_run = gen_specs['user']['dry_run']
    time_limit = gen_specs['user']['time_limit']
    epochs = app_args.split('epochs ')[-1]
    calc_status = UNSET_TAG

    H_o = np.zeros(1, dtype=gen_specs['out'])

    exctr = MPIExecutor.executor
    task = exctr.submit(app_name='ml_keras_mnist', num_procs=num_procs, app_args=app_args,
                        stderr='err.txt', hyperthreads=True, dry_run=dry_run,
                        wait_on_run=True)

    if not dry_run:
        poll_interval = 10  # secs
        while(not task.finished):
            if task.runtime > time_limit:
                task.kill()  # Timeout
                calc_status = WORKER_KILL_ON_TIMEOUT
            if exctr.manager_signal == 'finish':
                task.kill()
                calc_status = MAN_SIGNAL_FINISH
            elif exctr.manager_signal == 'kill':
                task.kill()
                calc_status = MAN_SIGNAL_KILL
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
        model_file = glob.glob('final_model_*')
        assert len(model_file), \
            "Keras Application did not write final output to file."
        assert f'Epoch {epochs}/{epochs}' in task.read_stdout(), \
            "Keras Application did not complete all epochs."

    current_dir = os.getcwd().split('/')[-1]  # gen_dir
    if dry_run:
        model_file = ['test.txt']

    H_o['cstat'] = calc_status
    H_o['model_file'] = os.path.join(current_dir, model_file[0])

    return H_o, persis_info, calc_status
