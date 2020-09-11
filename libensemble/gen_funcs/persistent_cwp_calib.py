import numpy as np
# from libensemble.gen_funcs.sampling import uniform_random_sample
from gemulator.emulation import emulation_prediction, emulation_builder, emulation_draws
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg
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


def select_next_theta(model, cur_thetas, obs, errstd, n_explore_theta, expect_impr_exit, persis_info):
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

    # Initialize output
    H_o = np.zeros((n_test_thetas + 1) * n_x, dtype=gen_specs['out'])

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
    H_o['quantile'] = np.quantile(test_fevals, 0.95)
    # MC Note: need to generate random / quantile-based failures
    # quantile = np.quantile(test_fevals, 0.95)
    # H_o['quantile'] = quantile

    # Generate initial batch of thetas
    theta, persis_info = gen_thetas(n_thetas, persis_info)
    for i, t in enumerate(theta):
        offset = i*n_x
        H_o['x'][offset:offset+n_x] = x
        H_o['thetas'][offset:offset+n_x] = t

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)
    # -------------------------------------------------------------------------

    # count = 0  # test only
    model_exists = False
    fevals = None
    future = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # count += 1  # test
        # print('count is', count,flush=True)
        if fevals is None:
            fevals = np.reshape(calc_in['f'], (n_thetas, n_x))
            failures = np.reshape(calc_in['failures'], (n_thetas, n_x))
            rebuild = True  # Currently only applies when async
        else:
            new_fevals = np.reshape(calc_in['f'], (n_thetas, n_x))
            new_failures = np.reshape(calc_in['failures'], (n_thetas, n_x))

            # SH Note: Presuming model input is everything so far.
            fevals = np.vstack((fevals, new_fevals))
            failures = np.vstack((failures, new_failures))
            rebuild = False  # Currently only applies when async

        # SH Testing. Cumulative failure rate
        frate = np.count_nonzero(failures)/failures.size
        print('failure rate is {}'.format(frate))

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

            if rebuild:
                executor = concurrent.futures.ThreadPoolExecutor()
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

        H_o['quantile'] = [np.inf]
        for i, t in enumerate(new_theta):
            offset = i*n_x
            H_o['x'][offset:offset+n_x] = x
            H_o['thetas'][offset:offset+n_x] = t

        tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)

    if async_build:
        try:
            executor.shutdown(wait=True)
        except Exception:
            pass

    return H, persis_info, FINISHED_PERSISTENT_GEN_TAG
