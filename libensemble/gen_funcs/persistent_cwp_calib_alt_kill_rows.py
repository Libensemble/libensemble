import numpy as np
# from libensemble.gen_funcs.sampling import uniform_random_sample
from gemulator.emulation import emulation_prediction, emulation_builder, emulation_draws
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


def gen_thetas(n, persis_info):
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


def gen_new_thetas(n, persis_info):
    thetas, _ = gen_thetas(n, persis_info)
    return thetas, persis_info


def select_next_theta(model, cur_thetas, obs, errstd, n_explore_theta, expect_impr_exit, persis_info):   # !!! add step_add_theta
    new_thetas, persis_info = gen_new_thetas(n_explore_theta, persis_info)

    fpred, _ = emulation_prediction(model, cur_thetas)
    fdraws = emulation_draws(model, new_thetas, options={'numsamples': 500})

    cur_chi2 = np.sum(((fpred - obs) / errstd) ** 2, axis=1)
    best_chi2 = np.min(cur_chi2)

    new_chi2 = np.zeros((fdraws.shape[0], fdraws.shape[2]))
    for k in np.arange(fdraws.shape[2]):
        new_chi2[:, k] = np.sum(((fdraws[:, :, k] - obs) / errstd) ** 2, axis=1)

    expect_improvement = ((best_chi2 > new_chi2)*(best_chi2 - new_chi2)).mean(axis=1)
    # prob_improvement = (best_chi2 > new_chi2).mean(axis=1)
    print('No. of thetas = {:d}'.format(cur_thetas.shape[0]))
    print('MAX EI = {:.2f}'.format(np.max(expect_improvement)))
    print('Best chi^2 = {:.2f}'.format(best_chi2))

    if np.max(expect_improvement) < 0.001 * obs.shape[0]:  # > 0.95:  tolerance?
        stop_flag = True
        new_theta = None
    else:
        stop_flag = False
        new_theta = np.copy(new_thetas[np.argmax(expect_improvement)])
        # print(new_theta)
        # print('new chi2: {:.2f}'.format(np.min(new_chi2)))

    return np.atleast_2d(new_theta), stop_flag, persis_info


def build_emulator(theta, x, fevals, failures):
    """Build the emulator"""
    # import time; time.sleep(5)  # test delay
    model = emulation_builder(theta, x, fevals, failures)
    return model


def gen_observations(fevals, errstd_constant, randstream):
    n_x = fevals.shape[0]
    errstd = errstd_constant * fevals
    obs = fevals + randstream.normal(0, errstd, n_x).reshape((n_x))
    return obs, errstd


# SH. Test condition.
def cancel_condition(row):
    if -1 in row:
        return True
    return False


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
                data_status[r, i] = -2  # SH: For cancelled ??

    # Send only these fields to existing H row and it will slot in change.
    H_o = np.zeros(len(sim_ids_to_cancel), dtype=[('sim_id', int), ('cancel', bool)])
    H_o['sim_id'] = sim_ids_to_cancel
    H_o['cancel'] = True
    send_mgr_worker_msg(comm, H_o)
    # SH: data_status will be modified without return - but optional for clarity
    # return data_status


def testcalib(H, persis_info, gen_specs, libE_info):
    """Gen to implement trainmseerror."""
    comm = libE_info['comm']
    randstream = persis_info['rand_stream']
    n_test_thetas = gen_specs['user']['n_test_thetas']
    n_thetas = gen_specs['user']['n_init_thetas']
    n_x = gen_specs['user']['num_x_vals']  # Num of x points
    # mse_exit = gen_specs['user']['mse_exit']  # MSE threshold for exiting
    step_add_theta = gen_specs['user']['step_add_theta']  # No. of thetas to generate per step
    expect_impr_exit = gen_specs['user']['expect_impr_exit']  # Expected improvement exit value
    n_explore_theta = gen_specs['user']['n_explore_theta']  # No. of thetas to explore
    async_build = gen_specs['user']['async_build']  # Build emulator in background thread
    errstd_constant = gen_specs['user']['errstd_constant']  # Constant for gener
    batch_last_sim_id = gen_specs['user']['batch_to_sim_id']  # Last batch sim_id
    ignore_cancelled = gen_specs['user']['ignore_cancelled']  # Ignore cancelled in data_status (still puts in feval/failures)

    # Initialize output
    pre_count = (n_test_thetas + 1) * n_x
    H_o = np.zeros(pre_count, dtype=gen_specs['out'])

    # Initialize exit criterion
    # H_o['mse'] = 1

    # Could generate initial inputs here or read in
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

    # arbitrary priority
    priority = np.arange(n_x*n_thetas)
    np.random.shuffle(priority)
    H_o['priority'] = priority

    print(H_o['priority'])

    # Generate initial batch of thetas for emulator
    theta, persis_info = gen_thetas(n_thetas, persis_info)
    for i, t in enumerate(theta):
        offset = i*n_x
        H_o['x'][offset:offset+n_x] = x
        H_o['thetas'][offset:offset+n_x] = t

    # send_mgr_worker_msg(comm, H_o)  # MC Note: Using send results in "unable to unpack NoneType"
    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
    # -------------------------------------------------------------------------

    # count = 0  # test only
    model_exists = False
    fevals = None
    future = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # count += 1  # test
        # print('count is', count,flush=True)

        if fevals is None:  # initial batch
            print(max(calc_in['sim_id']))
            fevals = np.reshape(calc_in['f'], (n_thetas, n_x))
            failures = np.reshape(calc_in['failures'], (n_thetas, n_x))

            data_status = np.full_like(fevals, 1, dtype=int)
            data_status[failures] = -1

            rebuild = True  # Currently only applies when async
        else:
            # fevals = np.full((n_thetas, n_x), np.nan, dtype=float)
            # failures = np.full_like(fevals, False)
            # data_status = np.full_like(fevals, 0, dtype=int)  # 0: incomplete, 1: successfully completed, -1: failed

            sim_id = calc_in['sim_id']
            r, c = divmod(sim_id - pre_count, n_x)  # r, c are arrays if sim_id is an array
            n_max_incoming_row = np.max(r) - fevals.shape[0] + 1

            if n_max_incoming_row > 0:
                fevals = np.pad(fevals, ((0, n_max_incoming_row), (0, 0)), 'constant', constant_values=np.nan)
                failures = np.pad(failures, ((0, n_max_incoming_row), (0, 0)), 'constant', constant_values=1)
                data_status = np.pad(data_status, ((0, n_max_incoming_row), (0, 0)), 'constant', constant_values=0)

            fevals[r, c] = calc_in['f']
            # print('fevals',fevals[(gen_specs['user']['n_init_thetas']):,:])
            print('fevals',fevals[r, :])  # MC test
            failures[r, c] = calc_in['failures']
            # print(failures[r, :])  # MC test
            # print(failures[(gen_specs['user']['n_init_thetas']):,:])

            # Set data_status. Using -2 for cancelled entries.
            for i in np.arange(r.shape[0]):
                if ignore_cancelled and data_status[r[i], c[i]] == -2:
                    continue
                data_status[r[i], c[i]] = -1 if calc_in['failures'][i] else 1

            # test to ensure failure and cancel row
            if 25 in r:
                print('r is:', r,flush=True)
                for i in np.arange(r.shape[0]):
                    if data_status[r[i], c[i]] == 1:
                        data_status[r[i], c[i]] = -1

            print('data_status row {} b4 cancel is:  {}'.format(r, data_status[r[0]:]),flush=True)

            if cancel_condition(data_status[r, :]):
                cancel_row(pre_count, r, n_x, data_status, comm)  # Sends cancellation - updates data_status
            print('data_status row {} aft cancel is: {}'.format(r, data_status[r[0]:]),flush=True)

            # print(data_status[r, :])  # MC test
            # new_fevals = np.full((n_thetas, n_x), np.nan)
            # new_fevals = np.reshape(calc_in['f'], (n_thetas, n_x))
            # new_failures = np.reshape(calc_in['failures'], (n_thetas, n_x))

            # SH Note: Presuming model input is everything so far.
            # fevals = np.vstack((fevals, new_fevals))
            # failures = np.vstack((failures, new_failures))
            print(r, np.mean(data_status[r, :] != 0))
            if np.mean(data_status[r, :] != 0) > 0.5:  # MC: wait for data or not, check fill proportion of incomplete rows
                rebuild = True
            else:
                rebuild = False  # Currently only applies when async
                tag, Work, calc_in = get_mgr_worker_msg(comm)
                continue  # continue in while loop without going forward with selection etc.

        # SH Testing. Cumulative failure rate
        frate = np.count_nonzero(failures)/failures.size
        print('failure rate is {:.6f}'.format(frate))

        # print('shapes {} {} {} {}\n'.format(theta.shape, x.shape, fevals.shape, failures.shape), flush=True)
        # MC: if condition, rebuild

        if async_build:
            if model_exists:
                if future.done():
                    model = future.result()
                    print('\nNew emulator built', flush=True)
                    rebuild = True
                else:
                    print('Re-using emulator', flush=True)
            else:
                executor = concurrent.futures.ThreadPoolExecutor()

            if rebuild:
                print('shapes: ')
                print(theta.shape, x.shape, fevals.shape, failures.shape)
                future = executor.submit(build_emulator, theta, x, fevals, failures)
                if not model_exists:
                    model = future.result()
                    model_exists = True
        else:
            # Always rebuilds...
            model = build_emulator(theta, x, fevals, failures)

        print('model id is {}'.format(id(model)), flush=True)  # test line - new model?
        new_theta, stop_flag, persis_info = \
            select_next_theta(model, theta, obs, errstd, n_explore_theta, expect_impr_exit, persis_info)
        print('theta removed: ' + str(model['theta_ind_removed']))
        # Exit gen when mse reaches threshold
        # print('\n maximum expected improvement is {}'.format(), flush=True)
        if stop_flag:
            print('Reached threshold.', flush=True)
            print('Number of thetas in total: {:d}'.format(theta.shape[0]))
            break

        # MC: If mse not under threshold, send additional thetas to simfunc
        n_thetas = step_add_theta
        # new_thetas, persis_info = gen_thetas(n_thetas, persis_info)
        theta = np.vstack((theta, new_theta))

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
        # send_mgr_worker_msg(comm, H_o)  # MC Note: Using send results in "unable to unpack NoneType"

    if async_build:
        try:
            executor.shutdown(wait=True)
        except Exception:
            pass

    return H, persis_info, FINISHED_PERSISTENT_GEN_TAG
