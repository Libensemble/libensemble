#!/usr/bin/env python
"""
A small example of reading/writing from/to files for libEnsemble evaluation 
"""

import numpy as np

def branin(x1,x2):
    a = 1.0
    b = 5.1/(4.0*pow(np.pi,2.0))
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8.0*np.pi)

    # # Numerical decisions
    # x1 = model.float(-5.0,10.0)
    # x2 = model.float(0.0,15.0)

    f = a*(x2 - b*x1**2 + c*x1 -r)**2 + s*(1-t)*np.cos(x1) + s

    return f

if __name__ == "__main__":

    x = np.loadtxt('./x.in')
    assert len(x) == 2, "branin.py is two dimensional"

    f = branin(x[0],x[1])
    np.savetxt('f.out',[f])
