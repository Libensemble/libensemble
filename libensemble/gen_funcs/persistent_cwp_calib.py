"""
This module contains a simple calibration example of using libEnsemble with gemulator package.
"""
import numpy as np
from libensemble.gen_funcs.cwp_calib_fn_support import gen_xs, gen_thetas, gen_observations, standardize_f, gen_true_theta
from libensemble.gen_funcs.cwp_calib_model import model_builder
from libensemble.gen_funcs.cwp_calib_model import select_next_theta, obviate_pend_thetas
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg, get_mgr_worker_msg, send_mgr_worker_msg
import concurrent.futures


def build_emulator(theta, x, fevals, failures):
    """Build the emulator"""
    model = model_builder(theta, x, fevals, failures)
    return model


def select_condition(data_status):
    if 0 in data_status:
        return False
    else:
        return True


def rebuild_condition(data_status):
    if 0 in data_status:
        return False
    else:
        return True


def create_arrays(calc_in, n_thetas, n_x, obs, errstd):
    """Create 2D (point * rows) arrays fevals, failures and data_status from calc_in"""
    fevals = np.reshape(calc_in['f'], (n_thetas, n_x))
    fevals = standardize_f(fevals, obs, errstd)  # standardize fevals by obs and supplied std
    failures = np.reshape(calc_in['failures'], (n_thetas, n_x))
    data_status = np.full_like(fevals, 1, dtype=int)
    data_status[failures] = -1
    return fevals, failures, data_status


def update_arrays(fevals, failures, data_status, calc_in, pre_count, n_x, obs, errstd, ignore_cancelled):
    """Unpack from calc_in into 2D (point * rows) fevals, failures, data_status"""
    sim_id = calc_in['sim_id']
    r, c = divmod(sim_id - pre_count, n_x)  # r, c are arrays if sim_id is an array
    fevals[r, c] = standardize_f(calc_in['f'], obs, errstd, c)
    failures[r, c] = calc_in['failures']

    # Set data_status. Using -2 for cancelled entries.
    for i in np.arange(r.shape[0]):
        if ignore_cancelled and data_status[r[i], c[i]] == -2:
            continue
        data_status[r[i], c[i]] = -1 if calc_in['failures'][i] else 1


def cancel_row(pre_count, r, n_x, data_status, comm):
    # Cancel rest of row
    sim_ids_to_cancel = []
    rows = np.unique(r)
    for r in rows:
        row_offset = r*n_x
        for i in range(n_x):
            sim_id_cancl = pre_count + row_offset + i
            if data_status[r, i] == 0:
                sim_ids_to_cancel.append(sim_id_cancl)
                data_status[r, i] = -2

    # Send only these fields to existing H row and it will slot in change.
    H_o = np.zeros(len(sim_ids_to_cancel), dtype=[('sim_id', int), ('cancel', bool)])
    H_o['sim_id'] = sim_ids_to_cancel
    H_o['cancel'] = True
    send_mgr_worker_msg(comm, H_o)


def load_H(H, x, thetas, offset=0):
    """Fill inputs num_points x num_thetas"""
    n_x = len(x)
    for i, t in enumerate(thetas):
        start = (i+offset)*n_x
        H['x'][start:start+n_x] = x
        H['thetas'][start:start+n_x] = t


def gen_testvals(ntests, x, persis_info, gen_specs):
    """Generate true values and test values using libE"""
    n_x = len(x)
    H_o = np.zeros((ntests + 1) * n_x, dtype=gen_specs['out'])
    H_o['quantile'] = [np.inf]  # Do not generate random failures

    # Generate true theta and load into H
    true_theta, persis_info = gen_true_theta(persis_info)
    H_o['x'][0:n_x] = x
    H_o['thetas'][0:n_x] = true_theta

    # Generate test thetas and load into H
    test_thetas, persis_info = gen_thetas(ntests, persis_info)
    load_H(H_o, x, test_thetas, 1)
    return H_o


def testcalib(H, persis_info, gen_specs, libE_info):
    """Gen to implement trainmseerror."""
    comm = libE_info['comm']
    randstream = persis_info['rand_stream']
    n_test_thetas = gen_specs['user']['n_test_thetas']
    n_thetas = gen_specs['user']['n_init_thetas']
    n_x = gen_specs['user']['num_x_vals']  # Num of x points
    step_add_theta = gen_specs['user']['step_add_theta']  # No. of thetas to generate per step
    n_explore_theta = gen_specs['user']['n_explore_theta']  # No. of thetas to explore
    async_build = gen_specs['user']['async_build']  # Build emulator in background thread
    errstd_constant = gen_specs['user']['errstd_constant']  # Constant for generator
    ignore_cancelled = gen_specs['user']['ignore_cancelled']  # Ignore cancelled in data_status (still puts in feval/failures)
    quantile = gen_specs['user']['quantile']  # Proportion of particles that succeed

    # Create points at which to evaluate the sim
    x, persis_info = gen_xs(n_x, persis_info)

    H_o = gen_testvals(n_test_thetas, x, persis_info, gen_specs)
    pre_count = len(H_o)

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
    if tag in [STOP_TAG, PERSIS_STOP]:
        return H, persis_info, FINISHED_PERSISTENT_GEN_TAG

    returned_fevals = np.reshape(calc_in['f'], (n_test_thetas + 1, n_x))
    true_fevals = returned_fevals[0, :]
    test_fevals = returned_fevals[1:, :]
    obs, errstd = gen_observations(true_fevals, errstd_constant, randstream)

    # Generate a batch of inputs and load into H
    H_o = np.zeros(n_x*(n_thetas), dtype=gen_specs['out'])
    threshold_to_failure = np.quantile(test_fevals, quantile)
    H_o['quantile'] = threshold_to_failure
    priority = np.arange(n_x*n_thetas)
    np.random.shuffle(priority)
    H_o['priority'] = priority

    theta, persis_info = gen_thetas(n_thetas, persis_info)
    load_H(H_o, x, theta)

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
    # -------------------------------------------------------------------------

    model_exists = False
    fevals = None
    future = None

    # store model_id and data_status used to build model
    model = None
    model_data_status = None
    future_model_data_status = None

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if fevals is None:  # initial batch
            fevals, failures, data_status = create_arrays(calc_in, n_thetas, n_x, obs, errstd)
            build_new_model = True
        else:
            # Update fevals, failures, data_status from calc_in
            update_arrays(fevals, failures, data_status, calc_in,
                           pre_count, n_x, obs, errstd, ignore_cancelled)

            if rebuild_condition(data_status):
                build_new_model = True
            else:
                build_new_model = False
                tag, Work, calc_in = get_mgr_worker_msg(comm)
                if tag in [STOP_TAG, PERSIS_STOP]:
                    break

        # Conditionally update model
        if async_build:
            if model_exists:
                if future.done():
                    if id(model) != id(future.result()):
                        print('\nNew emulator built', flush=True)
                    model = future.result()
                    model_data_status = np.copy(future_model_data_status)
                else:
                    print('Re-using emulator', flush=True)
            else:
                executor = concurrent.futures.ThreadPoolExecutor()

            if build_new_model:
                future_model_data_status = np.copy(data_status)
                future = executor.submit(build_emulator, theta, x, fevals, failures)
                if not model_exists:
                    model = future.result()
                    model_data_status = np.copy(future_model_data_status)
                    model_exists = True
        else:
            if build_new_model:
                model_data_status = np.copy(data_status)
                model = build_emulator(theta, x, fevals, failures)

        # Conditionally generate new thetas from model
        if select_condition(data_status):
            new_theta, stop_flag = \
                select_next_theta(model, theta, n_explore_theta, step_add_theta)

            if stop_flag:
                print('Reached threshold.', flush=True)
                print('Number of thetas in total: {:d}'.format(theta.shape[0]))
                break

            n_thetas = step_add_theta
            theta = np.vstack((theta, new_theta))

            data_status = np.pad(data_status, ((0, step_add_theta), (0, 0)), 'constant', constant_values=0)
            fevals = np.pad(fevals, ((0, step_add_theta), (0, 0)), 'constant', constant_values=np.nan)
            failures = np.pad(failures, ((0, step_add_theta), (0, 0)), 'constant', constant_values=1)

            H_o = np.zeros(n_x*(n_thetas), dtype=gen_specs['out'])

            # arbitrary priority
            priority = np.arange(n_x*n_thetas)
            np.random.shuffle(priority)
            H_o['priority'] = priority
            H_o['quantile'] = threshold_to_failure

            load_H(H_o, x, new_theta)
            tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)

        # Determine evaluations to cancel
        r_obviate = obviate_pend_thetas(model, theta, data_status)
        if r_obviate[0].shape[0] > 0:
            print('rows sent for cancel is:  {}'.format(r_obviate), flush=True)
            cancel_row(pre_count, r_obviate, n_x, data_status, comm)

    if async_build:
        try:
            executor.shutdown(wait=True)
        except Exception:
            pass

    return H, persis_info, FINISHED_PERSISTENT_GEN_TAG
