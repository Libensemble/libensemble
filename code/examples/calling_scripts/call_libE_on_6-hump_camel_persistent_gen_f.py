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

sys.path.append('../../src')
from libE import libE

from message_numbers import STOP_TAG 

def six_hump_camel(H, sim_out, obj_params):
    O = np.zeros(1,dtype=sim_out)

    # x1 = H['x'][i][0]
    # x2 = H['x'][i][1]
    # term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2;
    # term2 = x1*x2;
    # term3 = (-4+4*x2**2) * x2**2;

    # O['f'][i] = term1 + term2 + term3;

    # v = np.random.uniform(0,10)
    # print('About to sleep for :' + str(v))
    # time.sleep(v)
    O['f'] = 0.5*sum(H['x']*H['x'])
    O['grad'] = H['x']
    O['Hess_inv'] = np.eye(len(H['x']))
    
    return O

def persistent_Newton(g_in,gen_out,params,info):
    import ipdb; ipdb.set_trace()
    x = params['x0']

    while 1:
        D = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == STOP_TAG: break
                
        x = x - g_in['Hess_inv']*g_in['grad']

        O = np.zeros(1, dtype=gen_out)
        O['x'] = x
        O['priority'] = 1

    print(O)
    return O

n = 2

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': [six_hump_camel], # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float),
                     ('grad',float,n),
                     ('Hess_inv',float,(n,n))
                    ],
             'x0': np.array([1, 2]),
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': persistent_Newton,
             'in': ['grad','Hess_inv'],
             'out': [('x',float,n),
                     ('priority',float),
                    ],
             'num_inst': 1,
             'persistent': True,
             }

# Tell libEnsemble when to stop
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
    #     assert np.min(np.sum((H['x']-m)**2,1)) < tol

    #     print("\nlibEnsemble with APOSMM has identified the 6 minima within a tolerance " + str(tol))


