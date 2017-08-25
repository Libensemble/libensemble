# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html 
# 
# Execute via the following command:
#    mpiexec -np 4 python3 call_libE_on_GKLS.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
import numpy as np
from math import *

import time

sys.path.append('../../src')
from libE import libE

def six_hump_camel(H, sim_out, obj_params, info):
    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_out)


    for i,x in enumerate(H['x']):

        if 'blocking' in info:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] +  list(info['blocking'])
        else:
            ranks_involved = [MPI.COMM_WORLD.Get_rank()] 

        machinefilename = str(ranks_involved)

        with open(machinefilename,'w') as f:
            for rank in ranks_involved:
                b = obj_params['nodelist'][rank] + '\n'
                f.write(b*H['ranks_per_node'][i])

        outfile_name = "outfile_"+ machinefilename+".txt"
        if os.path.isfile(outfile_name) == True:
            os.remove(outfile_name)

        process = subprocess.call(["mpiexec","-np",H[i]['ranks_per_node']*len(ranks_involved),"-machinefile",machinefilename,python, "./helloworld.py"], stdout = open(outfile_name,'w'), shell=False)

        x1 = H['x'][i][0]
        x2 = H['x'][i][1]
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2;
        term2 = x1*x2;
        term3 = (-4+4*x2**2) * x2**2;

        O['f'][i] = term1 + term2 + term3;

        # v = np.random.uniform(0,10)
        # print('About to sleep for :' + str(v))
        # time.sleep(v)
    
    return O

def uniform_random_sample(g_in,gen_out,params,info):
    ub = params['ub']
    lb = params['lb']
    n = len(lb)

    if len(g_in) == 0: 
        b = params['initial_batch_size']

        O = np.zeros(b, dtype=gen_out)
        for i in range(0,b):
            x = np.random.uniform(lb,ub,(1,n))
            O['x'][i] = x
            O['num_nodes'][i] = 1
            O['ranks_per_node'][i] = 16
            O['priority'] = 1
        
    else:
        O = np.zeros(1, dtype=gen_out)
        O['x'] = len(g_in)*np.ones(n)
        O['num_nodes'] = np.random.choice([1,2,3,4]) 
        O['ranks_per_node'] = np.random.randint(1,17)

    return O


#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': [six_hump_camel], # This is the function whose output is being minimized
             'in': ['x','num_nodes','ranks_per_node'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'params': {'nodelist': ['b1','b1','b2','b3','b4']},
             # 'save_every_k': 10
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_random_sample,
             'in': ['sim_id'],
             'out': [('x',float,2),
                     ('priority',float),
                     ('num_nodes',int),
                     ('ranks_per_node',int),
                    ],
             'params': {'lb': np.array([-3,-2]),
                        'ub': np.array([ 3, 2]),
                        'initial_batch_size': 5,
                       },
             'num_inst': 1,
             'batch_mode': False,
             'give_all_with_same_priority': False,
             # 'save_every_k': 10
             }

# Tell LibEnsemble when to stop
exit_criteria = {'sim_max': 10}

np.random.seed(1)

# Perform the run
H = libE(sim_specs, gen_specs, exit_criteria)

if MPI.COMM_WORLD.Get_rank() == 0:
    filename = '6-hump_camel_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)


    # minima = np.array([[ -0.089842,  0.712656],
    #                    [  0.089842, -0.712656],
    #                    [ -1.70361,  0.796084],
    #                    [  1.70361, -0.796084],
    #                    [ -1.6071,   -0.568651],
    #                    [  1.6071,    0.568651]])
    # tol = 0.1
    # for m in minima:
    #     print(np.min(np.sum((H['x']-m)**2,1)))
    #     assert(np.min(np.sum((H['x']-m)**2,1)) < tol)

    #     print("\nLibEnsemble with APOSMM has identified the 6 minima within a tolerance " + str(tol))


