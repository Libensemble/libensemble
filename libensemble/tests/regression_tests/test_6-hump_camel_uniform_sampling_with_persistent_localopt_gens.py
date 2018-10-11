# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 {FILENAME}.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys, os             # for adding to path
import numpy as np

# Import libEnsemble main
from libensemble.libE import libE

# Import sim_func
from libensemble.sim_funcs.six_hump_camel import six_hump_camel

# Import gen_func
from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt

# Import alloc_func
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens


script_name = os.path.splitext(os.path.basename(__file__))[0]

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), ('grad',float,2) # This is the output from the function being minimized
                    ],
             }

gen_out = [('x',float,2),
      ('x_on_cube',float,2),
      ('priority',float),
      ('local_pt',bool),
      ('known_to_aposmm',bool), # Mark known points so fewer updates are needed.
      ('dist_to_unit_bounds',float),
      ('dist_to_better_l',float),
      ('dist_to_better_s',float),
      ('ind_of_better_l',int),
      ('ind_of_better_s',int),
      ('started_run',bool),
      ('num_active_runs',int),
      ('local_min',bool),
      ]

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_or_localopt,
             'in': [],
             'localopt_method':'LD_MMA',
             'xtol_rel':1e-4,
             'out': gen_out,
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': 2,
             'batch_mode': True,
             'num_active_gens':1,
             }

gen_out = [('x',float,2),
      ('x_on_cube',float,2),
      #('sim_id',int),
      ('priority',float),
      ('local_pt',bool),
      ('known_to_aposmm',bool), # Mark known points so fewer updates are needed.
      ('dist_to_unit_bounds',float),
      ('dist_to_better_l',float),
      ('dist_to_better_s',float),
      ('ind_of_better_l',int),
      ('ind_of_better_s',int),
      ('started_run',bool),
      ]

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 1000, 'elapsed_wallclock_time': 300}

np.random.seed(1)
persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}

alloc_specs = {'out':gen_out, 'alloc_f':start_persistent_local_opt_gens}
# Don't do a "persistent worker run" if only one wokrer
if MPI.COMM_WORLD.Get_size() == 2:
    quit()
# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    np.save(filename, H)

    minima = np.array([[ -0.089842,  0.712656],
                       [  0.089842, -0.712656],
                       [ -1.70361,  0.796084],
                       [  1.70361, -0.796084],
                       [ -1.6071,   -0.568651],
                       [  1.6071,    0.568651]])
    tol = 0.1
    for m in minima:
        assert np.min(np.sum((H['x']-m)**2,1)) < tol

    print("\nlibEnsemble with Uniform random sampling has identified the 6 minima within a tolerance " + str(tol))


