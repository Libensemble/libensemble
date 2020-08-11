# SH: Based on borehole problem - please change
# SH: Uses x and thetas but evals are independent of x.

import numpy as np

bounds = np.array([[63070, 115600],
                   [63.1, 116],
                   [990, 1110],
                   [700, 820],
                   [0, np.inf],  # Not sure if the physics have a more meaningful upper bound
                   [1, 1.2],  # Very low probability of being outside of this range
                   [9855, 12045],
                   [1120, 1680]])


def borehole(H, persis_info, sim_specs, _):
    """
    Wraps the borehole function
    """

    H_o = np.zeros(H.shape[0], dtype=sim_specs['out'])
    H_o['f'] = borehole_func(H)

    return H_o, persis_info


def borehole_func(H):
    """This evaluates the Borehole function for n-by-8 input
    matrix x, and returns the flow rate through the Borehole. (Harper and Gupta, 1983)
    input:

    Parameters
    ----------

    x: matrix of dimension (n, 8), where n is the number of input configurations:

        .. code-block::

        x[:,0]: Tu, transmissivity of upper aquifer (m^2/year)
        x[:,1]: Tl, transmissivity of lower aquifer (m^2/year)
        x[:,2]: Hu, potentiometric head of upper aquifer (m)
        x[:,3]: Hl, potentiometric head of lower aquifer (m)
        x[:,4]: r, radius of influence (m)
        x[:,5]: rw, radius of borehole (m)
        x[:,6]: Kw, hydraulic conductivity of borehole (m/year)
        x[:,7]: L, length of borehole (m)

    Returns
    -------

    vector of dimension (n, 1):
        flow rate through the Borehole (m^3/year)

    """

    thetas = H['thetas']

    assert np.all(thetas >= bounds[:, 0]) and np.all(thetas <= bounds[:, 1]), "Point not within bounds"

    axis = 1
    if thetas.ndim == 1:
        axis = 0

    (Tu, Tl, Hu, Hl, r, rw, Kw, L) = np.split(thetas, 8, axis)

    numer = 2 * np.pi * Tu * (Hu - Hl)
    denom1 = 2 * L * Tu / (np.log(r/rw) * rw**2 * Kw)
    denom2 = Tu / Tl

    return (numer / (np.log(r/rw) * (1 + denom1 + denom2))).reshape(-1)
