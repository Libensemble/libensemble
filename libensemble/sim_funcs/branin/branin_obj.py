"""
Calls the branin function. Default behavior uses the python function, but
uncommenting lines will write x.in to file, call branin.py, and then read f.out. 
"""
import numpy as np
import subprocess
import os
import time
from libensemble.sim_funcs.branin.branin import branin

def call_branin(H,persis_info,sim_specs,libE_info):
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

    return O, persis_info
