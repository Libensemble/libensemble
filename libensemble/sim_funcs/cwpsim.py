import numpy as np
import time
from libensemble.executors.executor import Executor
from libensemble.message_numbers import (UNSET_TAG, TASK_FAILED,
                                         MAN_SIGNAL_KILL, WORKER_DONE)

# bounds for (Tu, Tl, Hu, Hl, r, Kw, rw, L)
bounds = np.array([[63070, 115600],
                   [63.1, 116],
                   [990, 1110],
                   [700, 820],
                   [0, np.inf],  # Not sure if the physics have a more meaningful upper bound
                   [9855, 12045],
                   [0.05, 0.15],  # Very low probability of being outside of this range
                   [1120, 1680]])

bounds = np.array([[0, np.inf],
                   [0, np.inf],
                   [0, np.inf],
                   [0, np.inf],
                   [0, np.inf],
                   [0, np.inf],
                   [0, np.inf],
                   [0, np.inf]])

check_for_man_kills = True


def borehole_func(H):
    """This evaluates the Borehole function for n-by-8 input
    matrix x, and returns the flow rate through the Borehole. (Harper and Gupta, 1983)
    input:

    Parameters
    ----------
    theta: matrix of dimentsion (n, 6),
        theta[:,0]: Tu, transmissivity of upper aquifer (m^2/year)
        theta[:,1]: Tl, transmissivity of lower aquifer (m^2/year)
        theta[:,2]: Hu, potentiometric head of upper aquifer (m)
        theta[:,3]: Hl, potentiometric head of lower aquifer (m)
        theta[:,4]: r, radius of influence (m)
        theta[:,5]: Kw, hydraulic conductivity of borehole (m/year)

    x: matrix of dimension (n, 3), where n is the number of input configurations:
        .. code-block::
        x[:,0]: rw, radius of borehole (m)
        x[:,1]: L, length of borehole (m)
        x[:,2]: a in {0, 1}. type label for modification

    Returns
    -------

    vector of dimension (n, 1):
        flow rate through the Borehole (m^3/year)

    """

    thetas = H['thetas']
    xs = H['x']

    if not (np.all(thetas >= bounds[:6, 0]) and
            np.all(thetas <= bounds[:6, 1]) and
            np.all(xs[:, :-1] >= bounds[6:, 0]) and
            np.all(xs[:, :-1] <= bounds[6:, 1])):
        return np.nan

    # assert np.all(thetas >= bounds[:6, 0]) and \
    #     np.all(thetas <= bounds[:6, 1]) and \
    #     np.all(xs[:, :-1] >= bounds[6:, 0]) and \
    #     np.all(xs[:, :-1] <= bounds[6:, 1]), "Point not within bounds"

    taxis = 1
    if thetas.ndim == 1:
        taxis = 0
    (Tu, Tl, Hu, Hl, r, Kw) = np.split(thetas, 6, taxis)

    xaxis = 1
    if xs.ndim == 1:
        xaxis = 0
    (rw, L) = np.split(xs[:, :-1], 2, xaxis)

    numer = 2 * np.pi * Tu * (Hu - Hl)
    denom1 = 2 * L * Tu / (np.log(r/rw) * rw**2 * Kw)
    denom2 = Tu / Tl

    f = (numer / (np.log(r/rw) * (1 + denom1 + denom2))).reshape(-1)

    f[xs[:, -1] == 1] = f[xs[:, -1].astype(bool)] ** (1.5)

    return f


def polling_loop(exctr, task, sim_id):
    """ Poll task for complettion and for manager kill signal"""
    calc_status = UNSET_TAG
    poll_interval = 0.01

    # Poll task for finish and poll manager for kill signals
    while(not task.finished):
        exctr.manager_poll()
        if exctr.manager_signal == 'kill':
            task.kill()
            calc_status = MAN_SIGNAL_KILL
            print('Manager killed sim_id {} task.state {}'.format(sim_id, task.state), flush=True)
            break
        else:
            task.poll()
            time.sleep(poll_interval)

    if task.state == 'FAILED':
        calc_status = TASK_FAILED  # Failed - e.g. due to bounds check failure

    return calc_status


def add_delay(subp_opts, sim_id):
    """Add delay to borehole calculation to give chance to kill

    For testing - make one point per row take longer.
    """
    delay = 0
    if not subp_opts['delay']:
        return delay
    if sim_id > subp_opts['delay_start']:
        if (sim_id + 1) % subp_opts['num_x'] == 0:
            delay = 3 + np.random.normal(scale=0.5)
            print('sim_id {} delay {}'.format(sim_id, delay), flush=True)
    return delay


def subproc_borehole_func(H, subp_opts, libE_info):
    """This evaluates the Borehole function using a subprocess
    running compiled code.

    Note that the Executor base class submit runs a
    serial process in-place. This should work on compute nodes
    so long as there are free contexts.

    """
    sim_id = libE_info['H_rows'][0]
    delay = add_delay(subp_opts, sim_id)

    with open('input', 'w') as f:
        H['thetas'][0].tofile(f)
        H['x'][0].tofile(f)
        bounds.tofile(f)

    exctr = Executor.executor
    args = 'input' + ' ' + str(delay)

    task = exctr.submit(app_name='borehole', app_args=args, stdout='out.txt', stderr='err.txt')
    calc_status = polling_loop(exctr, task, sim_id)

    if calc_status in [MAN_SIGNAL_KILL, TASK_FAILED]:
        f = np.nan
    else:
        f = float(task.read_stdout())
        if subp_opts['check']:  # For debugging
            ftest = borehole_func(H)
            assert np.isclose(f, ftest), \
                "Subprocess f {} does not match in-line function {}".format(f, ftest)
    return f, calc_status


def borehole(H, persis_info, sim_specs, libE_info):
    """
    Wraps the borehole function
    """
    subprocess_borehole = sim_specs['user']['subprocess_borehole']
    H_o = np.zeros(H.shape[0], dtype=sim_specs['out'])

    # Subprocess to check kills
    if subprocess_borehole:
        subp_opts = sim_specs['user']['subp_opts']
        H_o['f'], calc_status = subproc_borehole_func(H, subp_opts, libE_info)
    else:
        calc_status = UNSET_TAG  # Calc_status gets printed in libE_stats.txt
        H_o['f'] = borehole_func(H)
        if H_o['f'] == np.nan:
            calc_status = TASK_FAILED

    if calc_status == UNSET_TAG:
        if H_o['f'] > H['quantile'][0]:
            H_o['failures'] = 1
            calc_status = TASK_FAILED
        else:
            H_o['failures'] = 0
            calc_status = WORKER_DONE

    return H_o, persis_info, calc_status
