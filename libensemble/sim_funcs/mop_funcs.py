# """
import numpy as np
import math


# Concave multiobjective function with the shape of the unit sphere in the
# first orthant. Credit to Deb et al. [2002].
def dtlz2(x, f):
    d = len(x)
    p = len(f)
    # Compute the kernel function g(x)
    gx = np.dot(x[p-1:d]-0.5, x[p-1:d]-0.5)
    # Compute the objectives
    f[0] = (1.0 + gx)
    for y in x[:p-1]:
        f[0] *= math.cos(math.pi * y / 2.0)
    for i in range(1, p):
        f[i] = (1.0 + gx) * math.sin(math.pi * x[p-1-i] / 2.0)
        for y in x[:p-1-i]:
            f[i] *= math.cos(math.pi * y / 2.0)
    return


# Convex multiobjective function with parabolic Pareto front
def convex_mop(x, f):
    d = len(x)
    p = len(f)
    for i in range(p):
        e = np.zeros(d)
        e[i] = 0.5
        f[i] = np.dot(x-e, x-e)
    return
