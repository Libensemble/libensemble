"""Placeholder for default doc """
from __future__ import division
from __future__ import absolute_import

import numpy as np
import subprocess
import os
import time
from branin import branin

# @profile
def call_branin(H,gen_info,sim_specs,libE_info):
    del libE_info # Ignored parameter

    """ Evaluates the Branin function """

    batch = len(H['x'])

    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):
        # Uncomment the following if you want to use the file system to do evaluations
        # devnull = open(os.devnull, 'w')
        # np.savetxt('./x.in', x, fmt='%16.16f', delimiter=' ', newline=" ")
        # p = subprocess.call(['python', 'branin.py'], cwd='./', stdout=devnull)
        # O['f'][i] = np.loadtxt('./f.out',dtype=float)

        O['f'][i] = branin(x[0],x[1])

        if 'uniform_random_pause_ub' in sim_specs: 
            time.sleep(sim_specs['uniform_random_pause_ub']*np.random.uniform())

        # if not H['local_pt'][i]:
        #     if np.random.uniform(0,1) < 0.1:
        #         print('blam')
        #         O['f'][i] = np.nan

    return O, gen_info
