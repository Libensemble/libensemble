"""Placeholder for default doc """
from __future__ import division
from __future__ import absolute_import

import numpy as np
import subprocess
import os
import time

def call_GKLS(H,sim_out,obj_params,info):
    """ Evaluates GKLS problem (f) in dimension (d) with (n) local minima
    
    Since we currently copy the directory for each rank, each function is
    evaluated with rank=0.
    """
    d = obj_params['problem_dimension']
    p_num = obj_params['problem_number']
    num_min = obj_params['number_of_minima'] 

    batch = len(H['x'])

    O = np.zeros(batch,dtype=sim_out)

    for i,x in enumerate(H['x']):
        devnull = open(os.devnull, 'w')
        np.savetxt('./x0000.in', x, fmt='%16.16f', delimiter=' ', newline=" ")
        p = subprocess.call(['./gkls_single','-d',str(d),'-n',str(num_min),'-f',str(p_num),'-r','0'], cwd='./', stdout=devnull)

        O['f'][i] = np.loadtxt('./f0000.out',dtype=float)

        if 'uniform_random_pause_ub' in obj_params: 
            time.sleep(obj_params['uniform_random_pause_ub']*np.random.uniform())

        # if not H['local_pt'][i]:
        #     if np.random.uniform(0,1) < 0.1:
        #         print('blam')
        #         O['f'][i] = np.nan

    return O
