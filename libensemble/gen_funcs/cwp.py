import numpy as np
from libensemble.gen_funcs.sampling import uniform_random_sample
from gemulator.emulation import emulation_prediction, emulation_builder, emulation_draws

# To obtain cwp test data (modify <prefix>)
# cwp_dir = '<prefix>/calibration'
# train_dir = os.path.join(cwp_dir, 'functionbase/examples/output/debug/.tmperror/trainmseerror')
# theta = np.loadtxt(os.path.join(train_dir, 'traintheta.csv'),delimiter=',',ndmin=2)
# x = np.loadtxt(os.path.join(train_dir, 'trainx.csv'),delimiter=',',ndmin=2, dtype=object)
# fevals = np.loadtxt(os.path.join(train_dir, 'trainfevals.csv'),delimiter=',',ndmin=2)
# failures =  np.loadtxt(os.path.join(train_dir, 'trainfailures.csv'),delimiter=',',ndmin=2)


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

    rw = randstream.normal(1.1, 0.0161812, n)
    L = randstream.uniform(1120, 1680, n)

    cat = np.repeat(0, n)
    cat[randstream.uniform(0, 1, n) >= 0.5] = 1

    x = np.column_stack((rw, L))
    xs = np.empty((n, x.shape[1]+1), dtype='object')
    xs[:, :x.shape[1]] = x
    xs[:, -1] = cat

    return xs, persis_info


def trainmseerror(H, persis_info, gen_specs, libE_info):
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
        print(gen_specs['out'])

        # (x, thetas)
        # for i, x in enumerate(xs):
        #     offset = i*n_thetas
        #     H_o['x'][offset:offset+n_thetas] = x
        #     H_o['thetas'][offset:offset+n_thetas] = thetas

        # !!! Populating through H_o
        # (thetas, x)
        for i, t in enumerate(thetas):
            offset = i*n_x
            H_o['x'][offset:offset+n_x] = xs
            H_o['thetas'][offset:offset+n_x] = t

        # If use persistent gen can store x and thetas here...
        persis_info['x'] = xs  # tied to worker
        persis_info['thetas'] = thetas  # tied to worker

        return H_o, persis_info
    else:
        # Note currently each x will evaluate the same as x is independent in sim - but helps check values.
        failures = np.zeros((n_thetas, n_x))
        x = persis_info['x']
        theta = persis_info['thetas']
        fevals = np.reshape(H['f'], (n_thetas, n_x))

        # Commented lines from original file.
        # N = fevals.shape[0]
        # Ntrain = int(N / 10)
        # print(fevals.shape)
        # indices = np.arange(N)
        # np.random.seed(0)  # for reproducible results
        # # np.random.shuffle(indices)
        # train_indices = indices[:Ntrain]
        # test_indices = indices[Ntrain:]

        # train model
        # print(np.sum(failures, axis=0))
        # print(np.sum(failures, axis=1))

        model = emulation_builder(theta, x, fevals, failures)

        predtrain, predvar = emulation_prediction(model, theta)
        trainmse = np.nanmean((predtrain[model['theta_ind_valid'], :] -
                              fevals[model['theta_ind_valid'], :]) ** 2)

        # draws = emulation_draws(model, theta)

        # H_o = ?

        return H_o, persis_info
