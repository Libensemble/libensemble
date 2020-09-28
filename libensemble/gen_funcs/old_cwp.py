import numpy as np
# from libensemble.gen_funcs.sampling import uniform_random_sample
from gemulator.emulation import emulation_prediction, emulation_builder, emulation_draws


# SH: Based on borehole problem - please change.
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
    n_thetas = gen_specs['user']['n_thetas']
    n_x = gen_specs['user']['gen_batch_size']  # Num of x points

    # Could generate initial inputs here or read in
    if H.size == 0:
        print('Generating initial inputs')
        # H_x, persis_info = uniform_random_sample(H, persis_info, gen_specs, libE_info)
        xs, persis_info = gen_xs(n_x, persis_info)
        thetas, persis_info = gen_thetas(n_thetas, persis_info)

        H_o = np.zeros(n_x*n_thetas, dtype=gen_specs['out'])

        # Initialize exit criterion
        H_o['mse'] = 1

        # (x, thetas)
        # for i, x in enumerate(xs):
        #     offset = i*n_thetas
        #     H_o['x'][offset:offset+n_thetas] = x
        #     H_o['thetas'][offset:offset+n_thetas] = thetas

        # (thetas, x)
        for i, t in enumerate(thetas):
            offset = i*n_x
            H_o['x'][offset:offset+n_x] = xs
            H_o['thetas'][offset:offset+n_x] = t

        # If use persistent gen can store x and thetas here...
        persis_info['x'] = xs  # tied to worker
        persis_info['thetas'] = thetas  # tied to worker

        # Store a fixed test dataset in persistent info
        n_test_thetas = 100

        test_thetas, persis_info = gen_thetas(n_test_thetas, persis_info)
        persis_info['test_thetas'] = test_thetas

        test_fevals = np.zeros(n_x * n_test_thetas, dtype=float)
        # MC: How do I ask simfunc to populate the test dataset and save in persis_info?
        persis_info['test_fevals'] = test_fevals

        return H_o, persis_info
    else:
        failures = np.zeros((n_thetas, n_x))  # MC Note: need to generate random / quantile-based failures
        x = persis_info['x']
        theta = persis_info['thetas']
        fevals = np.reshape(H['f'], (n_thetas, n_x))



        # MC: Goal - Call builder in initialization,
        # Call updater in subsequent loops
        model = emulation_builder(theta, x, fevals, failures)

        # predtrain, _ = emulation_prediction(model, theta)
        # trainmse = np.nanmean((predtrain[model['theta_ind_valid'], :] -
        #                       fevals[model['theta_ind_valid'], :]) ** 2)

        test_thetas = persis_info['test_thetas']
        predtest, _ = emulation_prediction(model, test_thetas)

        test_fevals = persis_info['test_fevals'].reshape(predtest.shape)
        mse = np.nanmean((predtest - test_fevals) ** 2)

        H['mse'] = mse

        # MC: If mse not under threshold, send one additional theta to simfunc
        n_new_thetas = 1
        new_thetas, persis_info = gen_thetas(n_new_thetas, persis_info)
        # Update theta here (?)
        persis_info['thetas'] = np.vstack((theta, new_thetas))

        return H, persis_info
