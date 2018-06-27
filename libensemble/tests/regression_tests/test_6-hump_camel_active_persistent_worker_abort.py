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
#sys.path.append('../../src')
from libensemble.libE import libE

# Import sim_func 
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs'))
from libensemble.sim_funcs.six_hump_camel import six_hump_camel

# Import gen_func 
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/gen_funcs'))
from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt

# Import alloc_func 
#sys.path.append(os.path.join(os.path.dirname(__file__), '../../examples/alloc_funcs'))
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens


script_name = os.path.splitext(os.path.basename(__file__))[0]

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': six_hump_camel, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             }

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': uniform_or_localopt,
             'in': [],
             'localopt_method':'LN_BOBYQA',
             'xtol_rel':1e-4,
             'out': [('x_on_cube',float,2),
                     ('x',float,2),
                     ('dist_to_unit_bounds',float),
                     ('dist_to_better_l',float),
                     ('dist_to_better_s',float),
                     ('ind_of_better_l',int),
                     ('ind_of_better_s',int),
                     ('local_pt',bool),
                     ('num_active_runs',int),
                     ('local_min',bool),
                    ],
             'lb': np.array([-3,-2]),
             'ub': np.array([ 3, 2]),
             'gen_batch_size': 2,
             'batch_mode': True,
             'dist_to_bound_multiple': 0.5,
             'localopt_maxeval': 4,
             'num_inst':1,
             }

gen_out = [('x',float,2),
      ('x_on_cube',float,2),
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
      ]

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 10} # Intentially set low so as to test that a worker in persistent mode can be terminated correctly

np.random.seed(1)

alloc_specs = {'out':gen_out, 'alloc_f':start_persistent_local_opt_gens}
if MPI.COMM_WORLD.Get_size() == 2: 
    # Can't do a "persistent worker run" if only one worker
    quit()
# Perform the run
H, gen_info, flag = libE(sim_specs, gen_specs, exit_criteria, alloc_specs=alloc_specs)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
    short_name = script_name.split("test_", 1).pop()
    filename = short_name + '_results_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(MPI.COMM_WORLD.Get_size())
    print("\n\n\nRun completed.\nSaving results to file: " + filename)
    # if flag == 2:
    #     print("\n\n\nKilling COMM_WORLD")
    #     MPI.COMM_WORLD.Abort()
