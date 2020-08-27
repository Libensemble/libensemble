import numpy as np
# from libensemble.gen_funcs.sampling import uniform_random_sample
from gemulator.emulation import emulation_prediction, emulation_builder, emulation_draws
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg


# Test push from github online
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


def testmseerror(H, persis_info, gen_specs, libE_info):
    """Gen to implement trainmseerror."""
    comm = libE_info['comm']
    n_test_thetas = gen_specs['user']['n_test_thetas']
    n_thetas = gen_specs['user']['n_init_thetas']
    n_x = gen_specs['user']['num_x_vals']  # Num of x points
    mse_exit = gen_specs['user']['mse_exit']  # MSE threshold for exiting
    step_add_theta = gen_specs['user']['step_add_theta']  # No. of thetas to generate per step

    # Initialize output
    H_o = np.zeros(n_x*(n_test_thetas+n_thetas), dtype=gen_specs['out'])

    # Initialize exit criterion
    # H_o['mse'] = 1

    # Could generate initial inputs here or read in
    x, persis_info = gen_xs(n_x, persis_info)

    # Generate test thetas
    test_thetas, persis_info = gen_thetas(n_test_thetas, persis_info)
    offset = 0
    for t in test_thetas:
        H_o['x'][offset:offset+n_x] = x
        H_o['thetas'][offset:offset+n_x] = t
        offset += n_x

    # Generate initial batch of thetas
    theta, persis_info = gen_thetas(n_thetas, persis_info)
    for t in theta:
        H_o['x'][offset:offset+n_x] = x
        H_o['thetas'][offset:offset+n_x] = t
        offset += n_x

    tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)

    # count = 0  # test only
    test_fevals = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # count += 1  # test
        # print('count is', count,flush=True)

        if test_fevals is None:
            ntevals = n_x * n_test_thetas
            test_fevals = np.reshape(calc_in['f'][:ntevals], (n_test_thetas, n_x))
            fevals = np.reshape(calc_in['f'][ntevals:], (n_thetas, n_x))

            # MC Note: need to generate random / quantile-based failures
            failures = np.reshape(calc_in['failures'][ntevals:], (n_thetas, n_x))
        else:
            new_fevals = np.reshape(calc_in['f'], (n_thetas, n_x))
            new_failures = np.reshape(calc_in['failures'], (n_thetas, n_x))

            # SH Note: Presuming model input is everything so far.
            fevals = np.vstack((fevals, new_fevals))
            failures = np.vstack((failures, new_failures))

        # MC: Goal - Call builder in initialization,
        # Call updater in subsequent loops

        # print('shapes {} {} {} {}\n'.format(theta.shape, x.shape, fevals.shape, failures.shape), flush=True)
        model = emulation_builder(theta, x, fevals, failures)

        # predtrain, _ = emulation_prediction(model, theta)
        # trainmse = np.nanmean((predtrain[model['theta_ind_valid'], :] -
        #                       fevals[model['theta_ind_valid'], :]) ** 2)

        predtest, _ = emulation_prediction(model, test_thetas)
        mse = np.nanmean((predtest - test_fevals) ** 2)

        # H_o['mse'] = mse
        # Exit gen when mse reaches threshold
        print('\n mse is {}'.format(mse), flush=True)
        if mse < mse_exit:
            print('Gen exiting on mse', flush=True)
            break

        # MC: If mse not under threshold, send one additional theta to simfunc
        n_thetas = step_add_theta
        new_thetas, persis_info = gen_thetas(n_thetas, persis_info)
        theta = np.vstack((theta, new_thetas))

        H_o = np.zeros(n_x*(n_thetas), dtype=gen_specs['out'])
        for i, t in enumerate(new_thetas):
            offset = i*n_x
            H_o['x'][offset:offset+n_x] = x
            H_o['thetas'][offset:offset+n_x] = t

        tag, Work, calc_in = sendrecv_mgr_worker_msg(comm, H_o)

    return H, persis_info, FINISHED_PERSISTENT_GEN_TAG
