import numpy as np
import time
from libensemble.executors.executor import Executor
from libensemble.sim_funcs.surmise_test_function import borehole_true
from libensemble.message_numbers import (UNSET_TAG, TASK_FAILED,
                                         MAN_SIGNAL_KILL, WORKER_DONE)


def polling_loop(exctr, task):
    """ Poll task for complettion and for manager kill signal"""
    calc_status = UNSET_TAG
    poll_interval = 0.01

    # Poll task for finish and poll manager for kill signals
    while(not task.finished):
        exctr.manager_poll()
        if exctr.manager_signal == 'kill':
            task.kill()
            calc_status = MAN_SIGNAL_KILL
            break
        else:
            task.poll()
            time.sleep(poll_interval)

    if task.state == 'FINISHED':
        calc_status = WORKER_DONE
    elif task.state == 'FAILED':
        calc_status = TASK_FAILED  # If run actually fails for some reason

    return calc_status


def subproc_borehole(H, delay):
    """This evaluates the Borehole function using a subprocess
    running compiled code.

    Note that the Executor base class submit runs a
    serial process in-place. This should work on compute nodes
    so long as there are free contexts.

    """
    with open('input', 'w') as f:
        H['thetas'][0].tofile(f)
        H['x'][0].tofile(f)

    exctr = Executor.executor
    args = 'input' + ' ' + str(delay)

    task = exctr.submit(app_name='borehole', app_args=args, stdout='out.txt', stderr='err.txt')
    calc_status = polling_loop(exctr, task)

    if calc_status in [MAN_SIGNAL_KILL, TASK_FAILED]:
        f = np.inf
    else:
        f = float(task.read_stdout())
    return f, calc_status


def borehole(H, persis_info, sim_specs, libE_info):
    """
    Wraps the borehole function
    Subprocess to test receiving kill signals from manager
    """
    calc_status = UNSET_TAG  # Calc_status gets printed in libE_stats.txt
    H_o = np.zeros(H['x'].shape[0], dtype=sim_specs['out'])

    # Add a delay so subprocessed borehole takes longer
    sim_id = libE_info['H_rows'][0]
    delay = 0
    if sim_id > sim_specs['user']['init_sample_size']:
        delay = 2 + np.random.normal(scale=0.5)

    f, calc_status = subproc_borehole(H, delay)

    # Failure model (excluding observations)
    if sim_id > sim_specs['user']['num_obs']:
        if (f / borehole_true(H['x'])) > 1.25:
            f = np.inf
            calc_status = TASK_FAILED
            print('Failure of sim_id {}'.format(sim_id), flush=True)

    H_o['f'] = f
    return H_o, persis_info, calc_status
