"""Contains parameter selection and obviation methods for cwpCalibration."""

import numpy as np
import scipy.stats as sps


class thetaprior:
    """Define the class instance of priors provided to the methods."""

    def lpdf(theta):
        """Return log prior density."""
        if theta.ndim > 1.5:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5), 1))
        else:
            return np.squeeze(np.sum(sps.norm.logpdf(theta, 1, 0.5)))

    def rnd(n):
        """Return random draws from prior."""
        return np.vstack((sps.norm.rvs(1, 0.5, size=(n, 4))))


def gen_true_theta():
    """Generate one parameter to be the true parameter for calibration."""
    theta0 = np.array([0.5] * 4)

    return theta0


def gen_thetas(n):
    """Generate and return n parameters for the test function."""
    thetas = thetaprior.rnd(n)
    return thetas


def gen_xs(nx, persis_info):
    """Generate and returns n inputs for the modified Borehole function."""
    randstream = persis_info['rand_stream']

    xs = randstream.uniform(0, 1, (nx, 3))
    xs[:, 2] = xs[:, 2] > 0.5

    return xs, persis_info


def gen_observations(fevals, errstd_constant, persis_info):
    """Generate observations."""
    randstream = persis_info['rand_stream']
    n_x = fevals.shape[0]
    errstd = errstd_constant  # * np.ones(n_x)
    obs = fevals + randstream.normal(0, errstd, n_x).reshape((n_x))
    return obs, errstd
