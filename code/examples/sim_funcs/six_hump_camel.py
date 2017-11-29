from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import subprocess, os
import numpy as np

import time

def six_hump_camel_with_different_ranks_and_nodes(H, gen_info, sim_specs, libE_info):
    """
    Evaluates the six hump camel but also performs a system call (to show one
    way of evaluating a compiled simulation).
    """

    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):

        if 'blocking' in libE_info:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] +  list(libE_info['blocking'])
        else:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] 

        machinefilename = 'machinefile_for_sim_id=' + str(libE_info['H_rows'][i] )+ '_ranks='+'_'.join([str(r) for r in ranks_involved])

        with open(machinefilename,'w') as f:
            for rank in ranks_involved:
                b = sim_specs['nodelist'][rank] + '\n'
                f.write(b*H['ranks_per_node'][i])

        outfile_name = "outfile_"+ machinefilename+".txt"
        if os.path.isfile(outfile_name):
            os.remove(outfile_name)

        call_str = ["mpiexec","-np",str(H[i]['ranks_per_node']*len(ranks_involved)),"-machinefile",machinefilename,"python", os.path.join(os.path.dirname(__file__),"helloworld.py")]
        process = subprocess.call(call_str, stdout = open(outfile_name,'w'), shell=False)

        O['f'][i] = six_hump_camel_func(x)

        # v = np.random.uniform(0,10)
        # print('About to sleep for :' + str(v))
        # time.sleep(v)
    
    return O, gen_info


def six_hump_camel(H, gen_info, sim_specs, libE_info):
    """
    Evaluates the six_hump_camel_func and possible six_hump_camel_grad
    """
    del libE_info # Ignored parameter

    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H['x']):
        O['f'][i] = six_hump_camel_func(x)

        if 'grad' in O.dtype.names:
            O['grad'][i] = six_hump_camel_grad(x)

        if 'pause_time' in sim_specs:
            time.sleep(sim_specs['pause_time'])

    return O, gen_info


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

def six_hump_camel_grad(x):
    """
    Definition of the six-hump camel gradient
    """

    x1 = x[0]
    x2 = x[1]
    grad = np.zeros(2)

    grad[0] = 2.0*(x1**5 - 4.2*x1**3 + 4.0*x1 + 0.5*x2)
    grad[1] = x1 + 16*x2**3 - 8*x2

    return grad

