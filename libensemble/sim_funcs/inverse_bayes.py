# Sim_func

from __future__ import division
from __future__ import absolute_import

__all__ = ['likelihood_calculator']

import subprocess, os
import numpy as np

import time
import pdb

def likelihood_calculator(H, persis_info, sim_specs, libE_info):
    """
    Evaluates likelihood
    """
    del libE_info # Ignored parameter
    O = np.zeros(len(H['x']),dtype=sim_specs['out'])
    for i,x in enumerate(H['x']):
        O['like'][i] = six_hump_camel_func(x)

    return O, persis_info

def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2;
    term2 = x1*x2;
    term3 = (-4+4*x2**2) * x2**2;

    return  term1 + term2 + term3
