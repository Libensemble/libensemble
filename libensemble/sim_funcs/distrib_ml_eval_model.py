from libensemble.executors.mpi_executor import MPIExecutor
from libensemble.message_numbers import (
                                         MAN_SIGNAL_FINISH, WORKER_DONE,
                                         TASK_FAILED, WORKER_KILL_ON_TIMEOUT)
import os
import time
import glob
import numpy as np
import tensorflow as tf


__all__ = ['distrib_ml_eval_model']

def distrib_ml_eval_model(H, persis_info, sim_specs, libE_info):

    model_file = H['model_file']
    print('inside sim, ', model_file)
    H_o = np.zeros(1, dtype=sim_specs['out'])
    H_o['evaluation'] = 0.1234

    return H_o, persis_info

    # _ , (mnist_test_images, mnist_test_labels) = \
    #     tf.keras.datasets.mnist.load_data(path='mnist.npz')
    #
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (tf.cast(mnist_test_images[..., tf.newaxis] / 255.0, tf.float32),
    #              tf.cast(mnist_test_labels, tf.int64))
    # )

    # comm = libE_info['comm']
    # num_procs = gen_specs['user']['num_procs']
    # app_args = gen_specs['user']['app_args']
    # dry_run = gen_specs['user']['dry_run']
    # time_limit = gen_specs['user']['time_limit']
    # epochs = app_args.split('epochs ')[-1]
    #
    # exctr = MPIExecutor.executor
    # task = exctr.submit(calc_type='gen', num_procs=num_procs, app_args=app_args,
    #                     stdout='out.txt', stderr='err.txt', hyperthreads=True,
    #                     dry_run=dry_run)
    #
    # poll_interval = 30  # secs
    # while(not task.finished):
    #     if task.runtime > time_limit:
    #         task.kill()  # Timeout
    #         calc_status = WORKER_KILL_ON_TIMEOUT
    #     if exctr.manager_signal == 'finish':
    #         task.kill()
    #         calc_status = MAN_SIGNAL_FINISH
    #     else:
    #         time.sleep(poll_interval)
    #         task.poll()
    #         exctr.manager_poll(comm)
    #
    # if task.finished:
    #     if task.state == 'FINISHED':
    #         print("Task {} completed".format(task.name))
    #         calc_status = WORKER_DONE
    #     elif task.state == 'FAILED':
    #         print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
    #         calc_status = TASK_FAILED
    #     elif task.state == 'USER_KILLED':
    #         print("Warning: Task {} has been killed".format(task.name))
    #         calc_status = WORKER_KILL
    #     else:
    #         print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))
    #         calc_status = UNSET_TAG
    #
    # time.sleep(0.2)
    # model_file = glob.glob('./final*')
    # assert len(model_file),
    #     "Keras Application did not write final output to file."
    # assert f'Epoch {epochs}/{epochs}' in task.read_stdout(),
    #     "Keras Application did not complete all epochs."
    #
    # current_dir = os.getcwd().split('/')[-1]
    #
    # H_o = np.zeros(1, dtype=gen_specs['out'])
    # H_o['cstat'] = calc_status
    # H_o['model_file'] = current_dir + '/' + model_file[0]
    #
    # return H_o, persis_info, calc_status
