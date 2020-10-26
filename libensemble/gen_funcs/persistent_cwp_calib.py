import numpy as np
# from libensemble.gen_funcs.sampling import uniform_random_sample
from gemulator.emulation import emulation_builder
from gemulator.calibration_support import select_next_theta, obviate_pend_thetas
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg, get_mgr_worker_msg, send_mgr_worker_msg
import concurrent.futures


def gen_true_theta(persis_info):
    """Generate one parameter to be the true parameter for calibration."""
    randstream = persis_info['rand_stream']

    Tu = randstream.uniform(89000, 90000, 1)
    Tl = randstream.uniform(80, 100, 1)
    Hu = randstream.uniform(1030, 1070, 1)
    Hl = randstream.uniform(750, 770, 1)
    r = randstream.uniform(700, 900, 1)
    Kw = randstream.uniform(10800, 11100, 1)
    theta = np.column_stack((Tu, Tl, Hu, Hl, r, Kw))

    return theta, persis_info


def gen_thetas(n, persis_info, hard=True):
    """Generates and returns n parameters for the Borehole function, according to distributions
    outlined in Harper and Gupta (1983).

    input:
      n: number of parameter to generate
    output:
      matrix of (n, 6), input as to borehole_func(t, x) function
    """
    randstream = persis_info['rand_stream']

    Tu = randstream.uniform(63070, 115600, n)
    Tl = randstream.uniform(63.1, 116, n)
    Hu = randstream.uniform(990, 1110, n)
    Hl = randstream.uniform(700, 820, n)
    r = randstream.lognormal(7.71, 1.0056, n)
    if hard:
        Kw = randstream.uniform(1500, 15000, n)  # adding non-linearity to function
    else:
        Kw = randstream.uniform(9855, 12045, n)
    thetas = np.column_stack((Tu, Tl, Hu, Hl, r, Kw))
    return thetas, persis_info


def gen_xs(n, persis_info):
    """Generate and returns n inputs for the modified Borehole function."""
    randstream = persis_info['rand_stream']

    rw = randstream.normal(0.1, 0.0161812, n)
    L = randstream.uniform(1120, 1680, n)

    cat = np.repeat(0, n)
    cat[randstream.uniform(0, 1, n) >= 0.5] = 1

    x = np.column_stack((rw, L))
    xs = np.empty((n, x.shape[1] + 1), dtype='object')
    xs[:, :x.shape[1]] = x
    xs[:, -1] = cat

    return xs, persis_info


def build_emulator(theta, x, fevals, failures):
    """Build the emulator"""
    model = emulation_builder(theta, x, fevals, failures)
    return model


def standardize_f(fevals, obs, errstd, colind=None):
    if colind is None:
        return (fevals - obs) / errstd
    else:
        return (fevals - obs[colind]) / errstd[colind]


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


def gen_observations(fevals, errstd_constant, randstream):
    n_x = fevals.shape[0]
    errstd = errstd_constant * fevals
    obs = fevals + randstream.normal(0, errstd, n_x).reshape((n_x))
    return obs, errstd


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
    errstd_constant = gen_specs['user']['errstd_constant']  # Constant for gener
    ignore_cancelled = gen_specs['user']['ignore_cancelled']  # Ignore cancelled in data_status (still puts in feval/failures)

    # Initialize output
    pre_count = (n_test_thetas + 1) * n_x
    H_o = np.zeros(pre_count, dtype=gen_specs['out'])

    # Could generate initial inputs here
    x, persis_info = gen_xs(n_x, persis_info)

    # Generate true theta
    true_theta, persis_info = gen_true_theta(persis_info)
    test_thetas, persis_info = gen_thetas(n_test_thetas, persis_info)

    H_o['x'][0:n_x] = x
    H_o['thetas'][0:n_x] = true_theta

    for i, t in enumerate(test_thetas):
        offset = (i+1)*n_x
        H_o['x'][offset:offset+n_x] = x
        H_o['thetas'][offset:offset+n_x] = t

    H_o['quantile'] = [np.inf]

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
    if tag in [STOP_TAG, PERSIS_STOP]:
        return H, persis_info, FINISHED_PERSISTENT_GEN_TAG
    # -------------------------------------------------------------------------

    returned_fevals = np.reshape(calc_in['f'], (n_test_thetas + 1, n_x))
    true_fevals = returned_fevals[0, :]
    test_fevals = returned_fevals[1:, :]

    obs, errstd = gen_observations(true_fevals, errstd_constant, randstream)

    H_o = np.zeros(n_x*(n_thetas), dtype=gen_specs['out'])
    quantile = np.quantile(test_fevals, 0.95)
    H_o['quantile'] = quantile

    priority = np.arange(n_x*n_thetas)
    np.random.shuffle(priority)
    H_o['priority'] = priority

    # Generate initial batch of thetas for emulator
    theta, persis_info = gen_thetas(n_thetas, persis_info)
    for i, t in enumerate(theta):
        offset = i*n_x
        H_o['x'][offset:offset+n_x] = x
        H_o['thetas'][offset:offset+n_x] = t

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
            fevals = np.reshape(calc_in['f'], (n_thetas, n_x))
            fevals = standardize_f(fevals, obs, errstd)  # standardize fevals by obs and supplied std
            failures = np.reshape(calc_in['failures'], (n_thetas, n_x))

            data_status = np.full_like(fevals, 1, dtype=int)
            data_status[failures] = -1

            rebuild = True  # Currently only applies when async
        else:
            sim_id = calc_in['sim_id']
            r, c = divmod(sim_id - pre_count, n_x)  # r, c are arrays if sim_id is an array
            n_max_incoming_row = np.max(r) - fevals.shape[0] + 1

            if n_max_incoming_row > 0:
                fevals = np.pad(fevals, ((0, n_max_incoming_row), (0, 0)), 'constant', constant_values=np.nan)
                failures = np.pad(failures, ((0, n_max_incoming_row), (0, 0)), 'constant', constant_values=1)

            fevals[r, c] = standardize_f(calc_in['f'], obs, errstd, c)
            failures[r, c] = calc_in['failures']

            # Set data_status. Using -2 for cancelled entries.
            for i in np.arange(r.shape[0]):
                if ignore_cancelled and data_status[r[i], c[i]] == -2:
                    continue
                data_status[r[i], c[i]] = -1 if calc_in['failures'][i] else 1

            if rebuild_condition(data_status):
                rebuild = True
            else:
                rebuild = False  # Currently only applies when async
                tag, Work, calc_in = get_mgr_worker_msg(comm)

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

            if rebuild:
                future_model_data_status = np.copy(data_status)
                future = executor.submit(build_emulator, theta, x, fevals, failures)
                if not model_exists:
                    model = future.result()
                    model_data_status = np.copy(future_model_data_status)
                    model_exists = True
        else:
            # Always rebuilds...
            model_data_status = np.copy(model_data_status)
            model = build_emulator(theta, x, fevals, failures)

        if select_condition(data_status):
            new_theta, stop_flag = \
                select_next_theta(model, theta, n_explore_theta, step_add_theta)

            if stop_flag:
                print('Reached threshold.', flush=True)
                print('Number of thetas in total: {:d}'.format(theta.shape[0]))
                break

            n_thetas = step_add_theta
            theta = np.vstack((theta, new_theta))

            data_status = np.pad(data_status, ((0, step_add_theta), (0, 0)), constant_values=0)
            H_o = np.zeros(n_x*(n_thetas), dtype=gen_specs['out'])

            # arbitrary priority
            priority = np.arange(n_x*n_thetas)
            np.random.shuffle(priority)
            H_o['priority'] = priority

            H_o['quantile'] = quantile
            for i, t in enumerate(new_theta):
                offset = i*n_x
                H_o['x'][offset:offset+n_x] = x
                H_o['thetas'][offset:offset+n_x] = t

            tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
        else:
            pass

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
