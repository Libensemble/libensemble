"""
Created on Tue Feb  9 10:27:23 2021

@author: mosesyhc
"""
import numpy as np


def borehole(H, persis_info, sim_specs, libE_info):
    """
    Wraps the borehole function
    """

    H_o = np.zeros(H["x"].shape[0], dtype=sim_specs["out"])

    # If observation do not use failure model
    sim_id = libE_info["H_rows"][0]
    if sim_id > sim_specs["user"]["num_obs"]:
        H_o["f"] = borehole_failmodel(H["x"], H["thetas"])
    else:
        H_o["f"] = borehole_model(H["x"], H["thetas"])
    return H_o, persis_info


def borehole_failmodel(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""
    f = borehole_model(x, theta)
    wheretoobig = np.where((f / borehole_true(x)) > 1.25)
    f[wheretoobig[0], wheretoobig[1]] = np.inf
    return f


def borehole_model(x, theta):
    """Given x and theta, return matrix of [row x] times [row theta] of values."""
    theta = tstd2theta(theta)
    x = xstd2x(x)
    p = x.shape[0]
    n = theta.shape[0]

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    f = borehole_vec(x_stacked, theta_stacked).reshape((n, p))
    return f.T


def borehole_true(x):
    """Given x, return matrix of [row x] times 1 of values."""
    # assume true theta is [0.5]^d
    theta0 = np.atleast_2d(np.array([0.5] * 4))
    f0 = borehole_model(x, theta0)

    return f0


def borehole_vec(x, theta):
    """Given x and theta, return vector of values."""
    (Hu, Ld_Kw, Treff, powparam) = np.split(theta, theta.shape[1], axis=1)
    (rw, Hl) = np.split(x[:, :-1], 2, axis=1)
    numer = 2 * np.pi * (Hu - Hl)
    denom1 = 2 * Ld_Kw / rw**2
    denom2 = Treff

    f = ((numer / ((denom1 + denom2))) * np.exp(powparam * rw)).reshape(-1)

    return f


def tstd2theta(tstd, hard=True):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    if tstd.ndim < 1.5:
        tstd = tstd[:, None].T
    (Treffs, Hus, LdKw, powparams) = np.split(tstd, tstd.shape[1], axis=1)

    Treff = (0.5 - 0.05) * Treffs + 0.05
    Hu = Hus * (1110 - 990) + 990
    if hard:
        Ld_Kw = LdKw * (1680 / 1500 - 1120 / 15000) + 1120 / 15000
    else:
        Ld_Kw = LdKw * (1680 / 9855 - 1120 / 12045) + 1120 / 12045

    powparam = powparams * (0.5 - (-0.5)) + (-0.5)

    theta = np.hstack((Hu, Ld_Kw, Treff, powparam))
    return theta


def xstd2x(xstd):
    """Given standardized x in [0, 1]^2 x {0, 1}, return non-standardized x."""
    if xstd.ndim < 1.5:
        xstd = xstd[:, None].T
    (rws, Hls, labels) = np.split(xstd, xstd.shape[1], axis=1)

    rw = rws * (np.log(0.5) - np.log(0.05)) + np.log(0.05)
    rw = np.exp(rw)
    Hl = Hls * (820 - 700) + 700

    x = np.hstack((rw, Hl, labels))
    return x
