"""Placeholder for default doc """
from __future__ import division
from __future__ import absolute_import

import numpy as np
import subprocess
import os
import time

def call_GKLS_with_random_pause(x,obj_params):
    """ Evaluates GKLS problem (f) in dimension (d) with (n) local minima
    
    Since we currently copy the directory for each rank, each function is
    evaluated with rank=0.
    """
    d = obj_params['problem_dimension']
    f = obj_params['problem_number']
    n = obj_params['number_of_minima'] 

    devnull = open(os.devnull, 'w')
    np.savetxt('./x0000.in', x, fmt='%16.16f', delimiter=' ', newline=" ")
    p = subprocess.call(['./gkls_single','-d',str(d),'-n',str(n),'-f',str(f),'-r','0'], cwd='./', stdout=devnull)
    f = np.loadtxt('./f0000.out',dtype='float')

    time.sleep(obj_params['uniform_random_pause_ub']*np.random.uniform())
    # time.sleep(0.1)

    return f.flatten()

def call_GKLS(x,obj_params):
    """ Evaluates GKLS problem (f) in dimension (d) with (n) local minima
    
    Since we currently copy the directory for each rank, each function is
    evaluated with rank=0.
    """
    d = obj_params['problem_dimension']
    f = obj_params['problem_number']
    n = obj_params['number_of_minima'] 

    devnull = open(os.devnull, 'w')
    np.savetxt('./x0000.in', x, fmt='%16.16f', delimiter=' ', newline=" ")
    p = subprocess.call(['./gkls_single','-d',str(d),'-n',str(n),'-f',str(f),'-r','0'], cwd='./', stdout=devnull)
    f = np.loadtxt('./f0000.out',dtype='float')

    return f.flatten()
