"""
This module contains a simple calibration example of using libEnsemble with gemulator package.
"""
import numpy as np
from libensemble.gen_funcs.surmise_calib_support import gen_xs, gen_thetas, gen_observations, gen_true_theta, \
    thetaprior, select_next_theta, obviate_pend_theta
from surmise.calibration import calibrator
from surmise.emulation import emulator
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg, get_mgr_worker_msg, send_mgr_worker_msg
import concurrent.futures


def build_emulator(theta, x, fevals):
    """Build the emulator."""
    emu = emulator(x, theta, fevals, method='PCGPwM',
                   options={'xrmnan': 'all',
                            'thetarmnan': 'never',
                            'return_grad': True})
    emu.fit()
    return emu

def select_condition(pending):
    if 0 in pending:
        return False
    else:
        return True


def rebuild_condition(pending):  # needs changes
    if 0 in pending:
        return False
    else:
        return True


def create_arrays(calc_in, n_thetas, n_x):
    """Create 2D (point * rows) arrays fevals, failures and data_status from calc_in"""
    fevals = np.reshape(calc_in['f'], (n_x, n_thetas))
    pending = np.full(fevals.shape, False)
    complete = np.full(fevals.shape, True)

    return fevals, pending, complete


def pad_arrays(n_thetanew, n_x, fevals, pending, complete):
    fevals = np.hstack((fevals, np.full((n_x, n_thetanew), np.nan)))
    pending = np.hstack((pending, np.full((n_x, n_thetanew), True)))
    complete = np.hstack((complete, np.full((n_x, n_thetanew), False)))
    return


def update_arrays(fevals, pending, complete, calc_in, pre_count, n_x, ignore_cancelled):
    """Unpack from calc_in into 2D (point * rows) fevals, failures, data_status"""
    sim_id = calc_in['sim_id']
    r, c = divmod(sim_id - pre_count, n_x)  # r, c are arrays if sim_id is an array

    fevals[r, c] = calc_in['f']
    pending[r, c] = False
    complete[r, c] = True
    return


def cancel_columns(pre_count, r, n_x, pending, complete, comm):
    """Cancel columns"""
    sim_ids_to_cancel = []
    rows = np.unique(r)
    for r in rows:
        row_offset = r*n_x
        for i in range(n_x):
            sim_id_cancl = pre_count + row_offset + i
            if pending[r, i]:
                sim_ids_to_cancel.append(sim_id_cancl)
                pending[r, i] = 0

    # Send only these fields to existing H row and it will slot in change.
    H_o = np.zeros(len(sim_ids_to_cancel), dtype=[('sim_id', int), ('cancel', bool)])
    H_o['sim_id'] = sim_ids_to_cancel
    H_o['cancel'] = True
    send_mgr_worker_msg(comm, H_o)


def assign_priority(n_x, n_thetas):
    """Assign priorities to points."""
    # Arbitrary priorities
    priority = np.arange(n_x*n_thetas)
    np.random.shuffle(priority)
    return priority


def load_H(H, x, thetas, offset=0, set_priorities=False):
    """Fill inputs into H0.

    There will be num_points x num_thetas entries
    """
    n_x = len(x)
    for i, t in enumerate(thetas):
        start = (i+offset)*n_x
        H['x'][start:start+n_x] = x
        H['thetas'][start:start+n_x] = t

    if set_priorities:
        n_thetas = len(thetas)
        H['priority'] = assign_priority(n_x, n_thetas)


def gen_truevals(x, gen_specs):
    """Generate true values using libE."""
    n_x = len(x)
    H_o = np.zeros((1) * n_x, dtype=gen_specs['out'])

    # Generate true theta and load into H
    true_theta = gen_true_theta()
    H_o['x'][0:n_x] = x
    H_o['thetas'][0:n_x] = true_theta
    return H_o


def testcalib(H, persis_info, gen_specs, libE_info):
    """Gen to implement trainmseerror."""
    comm = libE_info['comm']
    randstream = persis_info['rand_stream']
    n_thetas = gen_specs['user']['n_init_thetas']
    n_x = gen_specs['user']['num_x_vals']  # Num of x points
    step_add_theta = gen_specs['user']['step_add_theta']  # No. of thetas to generate per step
    n_explore_theta = gen_specs['user']['n_explore_theta']  # No. of thetas to explore
    obsvar = gen_specs['user']['obsvar']  # Constant for generator
    ignore_cancelled = gen_specs['user']['ignore_cancelled']  # Ignore cancelled in data_status

    # Create points at which to evaluate the sim
    x, persis_info = gen_xs(n_x, persis_info)

    H_o = gen_truevals(x, gen_specs)
    pre_count = len(H_o)

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
    if tag in [STOP_TAG, PERSIS_STOP]:
        return H, persis_info, FINISHED_PERSISTENT_GEN_TAG

    returned_fevals = np.reshape(calc_in['f'], (1, n_x))
    true_fevals = returned_fevals
    obs, obsvar = gen_observations(true_fevals, obsvar, persis_info)

    # Generate a batch of inputs and load into H
    H_o = np.zeros(n_x*(n_thetas), dtype=gen_specs['out'])
    theta = gen_thetas(n_thetas)
    load_H(H_o, x, theta, set_priorities=True)
    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
    # -------------------------------------------------------------------------

    fevals = None
    prev_pending = None

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if fevals is None:  # initial batch
            fevals, pending, complete = create_arrays(calc_in, n_thetas, n_x)
            emu = build_emulator(theta, x, fevals)
            cal = calibrator(emu, obs, x, thetaprior, obsvar, method='directbayes')

            prev_pending = pending.copy()
            update_model = False
        else:
            # Update fevals, failures, data_status from calc_in
            update_arrays(fevals, pending, complete, calc_in,
                          pre_count, n_x, ignore_cancelled)

            update_model = rebuild_condition(pending, prev_pending)
            if not update_model:
                tag, Work, calc_in = get_mgr_worker_msg(comm)
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break

        if update_model:
            emu.update(theta=theta, f=fevals)
            cal.fit()

            prev_pending = pending.copy()
            update_model = False

        # Conditionally generate new thetas from model
        if select_condition(pending):
            new_theta, info = select_next_theta(step_add_theta, cal, emu, pending, n_explore_theta)

            # Add space for new thetas
            pad_arrays(len(new_theta), n_x, fevals, pending, complete)

            n_thetas = step_add_theta
            theta = np.vstack((theta, new_theta))
            H_o = np.zeros(n_x*(n_thetas), dtype=gen_specs['out'])
            load_H(H_o, x, new_theta, set_priorities=True)
            tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)

            # Determine evaluations to cancel
            pending, c_obviate = obviate_pend_theta(info, pending)
            if len(c_obviate) > 0:
                print('columns sent for cancel is:  {}'.format(c_obviate), flush=True)
                cancel_columns(pre_count, c_obviate, n_x, pending, complete, comm)

    return H, persis_info, FINISHED_PERSISTENT_GEN_TAG
