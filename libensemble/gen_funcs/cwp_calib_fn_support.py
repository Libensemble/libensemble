"""Contains parameter selection and obviation methods for cwpCalibration."""

import numpy as np


def gen_local_thetas(n, t0, rnge, r=0.1, mode='Gaussian'):
    """Generate new thetas close to given theta t0."""
    p = t0.shape[0] if t0.ndim == 1 else t0.shape[1]

    Z = np.random.normal(size=(n, p))
    thetas = t0 + r * rnge * Z

    return thetas


def gen_new_thetas(n, t0, rnge):
    thetas, _ = gen_local_thetas(n, t0, rnge)
    return thetas


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


def standardize_f(fevals, obs, errstd, colind=None):
    if colind is None:
        return (fevals - obs) / errstd
    else:
        return (fevals - obs[colind]) / errstd[colind]


def gen_observations(fevals, errstd_constant, randstream):
    n_x = fevals.shape[0]
    errstd = errstd_constant * fevals
    obs = fevals + randstream.normal(0, errstd, n_x).reshape((n_x))
    return obs, errstd
