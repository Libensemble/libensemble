from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import subprocess, os
import numpy as np

import time

def six_hump_camel_with_different_ranks_and_nodes(H_s, gen_info, sim_specs, info):
    batch = len(H_s['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H_s['x']):

        if 'blocking' in info:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] +  list(info['blocking'])
        else:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] 

        machinefilename = 'machinefile_for_sim_id=' + str(info['sim_id'][i] )+ '_ranks='+'_'.join([str(r) for r in ranks_involved])

        with open(machinefilename,'w') as f:
            for rank in ranks_involved:
                b = sim_specs['nodelist'][rank] + '\n'
                f.write(b*H_s['ranks_per_node'][i])

        outfile_name = "outfile_"+ machinefilename+".txt"
        if os.path.isfile(outfile_name):
            os.remove(outfile_name)

        call_str = ["mpiexec","-np",str(H_s[i]['ranks_per_node']*len(ranks_involved)),"-machinefile",machinefilename,"python", os.path.join(os.path.dirname(__file__),"helloworld.py")]
        process = subprocess.call(call_str, stdout = open(outfile_name,'w'), shell=False)

        O['f'][i] = six_hump_camel_func(H_s['x'][i])

        # v = np.random.uniform(0,10)
        # print('About to sleep for :' + str(v))
        # time.sleep(v)
    
    return O


def six_hump_camel(H_s, gen_info, sim_specs, info):
    batch = len(H_s['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])

    for i,x in enumerate(H_s['x']):
        O['f'][i] = six_hump_camel_func(H_s['x'][i])

        if 'grad' in O.dtype.names:
            O['grad'][i] = six_hump_camel_grad(H_s['x'][i])

        if 'pause_time' in sim_specs:
            time.sleep(sim_specs['pause_time'])

    return O


def six_hump_camel_func(x):
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2;
    term2 = x1*x2;
    term3 = (-4+4*x2**2) * x2**2;

    return  term1 + term2 + term3;

def six_hump_camel_grad(x):

    x1 = x[0]
    x2 = x[1]
    grad = np.zeros(2)

    grad[0] = 2.0*(x1**5 - 4.2*x1**3 + 4.0*x1 + 0.5*x2)
    grad[1] = x1 + 16*x2**3 - 8*x2

    return grad

