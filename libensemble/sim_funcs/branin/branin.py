#!/usr/bin/env python
"""
A small example of using the call_branin function from branin_obj.py in order
to (write to)/(read from) files when doing a sim_f evaluation
"""

import numpy as np


def branin(x1, x2):
    a = 1.0
    b = 5.1 / (4.0 * pow(np.pi, 2.0))
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    f = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    return f


if __name__ == "__main__":

    x = np.loadtxt("./x.in")
    assert len(x) == 2, "branin.py is two dimensional"

    f = branin(x[0], x[1])
    np.savetxt("f.out", [f])
