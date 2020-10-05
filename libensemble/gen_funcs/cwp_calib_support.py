"""Contains parameter selection and obviation methods for cwpCalibration."""

import numpy as np
from gemulator.calibration_support import gen_local_thetas


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
    # Kw = randstream.uniform(9855, 12045, n)
    Kw = randstream.uniform(1500, 15000, n)  # adding non-linearity to function
    thetas = np.column_stack((Tu, Tl, Hu, Hl, r, Kw))
    return thetas, persis_info


def gen_new_thetas(n, t0, rnge):
    thetas, _ = gen_local_thetas(n, t0, rnge)
    return thetas
