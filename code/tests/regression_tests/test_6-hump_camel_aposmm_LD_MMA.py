# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html 
# 
# Execute via the following command:
#    mpiexec -np 4 python3 call_6-hump_camel.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
sys.path.append('../../src')
from libE import libE

# Import sim_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs'))
from six_hump_camel import six_hump_camel

# Import gen_func 
sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from aposmm_logic import aposmm_logic

from math import gamma, pi, sqrt
script_name = os.path.splitext(os.path.basename(__file__))[0]

n = 2

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': [six_hump_camel], # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float),('grad',float,n), # This is the output from the function being minimized
                    ],
             }

gen_out = [('x',float,n),
      ('x_on_cube',float,n),
      ('sim_id',int),
      ('priority',float),
      ('local_pt',bool),
      ('known_to_aposmm',bool), # Mark known points so fewer updates are needed.
      ('dist_to_unit_bounds',float),
      ('dist_to_better_l',float),
      ('dist_to_better_s',float),
      ('ind_of_better_l',int),
      ('ind_of_better_s',int),
      ('started_run',bool),
      ('num_active_runs',int), # Number of active runs point is involved in
      ('local_min',bool),
      ('obj_component',int),
      ('pt_id',int), # To be used by APOSMM to identify points evaluated by different simulations
      ]

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': aposmm_logic,
             'in': [o[0] for o in gen_out] + ['f', 'grad', 'returned'],
             'out': gen_out,
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'initial_sample': 100,
             'localopt_method': 'LD_MMA',
             'rk_const': 0.5*((gamma(1+(n/2))*5)**(1/n))/sqrt(pi),
             'xtol_rel': 1e-2,
             'lhs_divisions':2,
             'batch_mode': True,
             'num_inst':1,
             }


# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 1000}

np.random.seed(1)

# Perform the run

for run in range(2):
    if run == 1:
        # Change the bounds to put a solution at a corner point (to test APOSMM's ability to give back a previously evaluated point)
        gen_specs['ub']= np.array([-2.9, -1.9])
        gen_specs['mu']= 1e-4
        exit_criteria['sim_max'] = 200

    H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria)

    if MPI.COMM_WORLD.Get_rank() == 0:
        short_name = script_name.split("test_", 1).pop()
        filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
        print("\n\n\nRun completed.\nSaving results to file: " + filename)
        np.save(filename, H)

        if run == 0:
            minima = np.array([[ -0.089842,  0.712656],
                               [  0.089842, -0.712656],
                               [ -1.70361,  0.796084],
                               [  1.70361, -0.796084],
                               [ -1.6071,   -0.568651],
                               [  1.6071,    0.568651]])
        else: 
            minima = np.array([[-2.9, -1.9]])

        tol = 1e-4
        for m in minima:
            print(np.min(np.sum((H['x']-m)**2,1)))
            assert np.min(np.sum((H['x']-m)**2,1)) < tol

        print("\nlibEnsemble with APOSMM using a gradient-based localopt method has identified the " + str(np.shape(minima)[0]) + " minima within a tolerance " + str(tol))


