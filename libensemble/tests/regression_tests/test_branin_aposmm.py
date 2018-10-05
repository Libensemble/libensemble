# """
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import sys             # for adding to path
import os
import numpy as np

# Import libEnsemble main
from libensemble.libE import libE

# Import sim_func and declare directory to be copied by each worker to do its evaluations in
import pkg_resources; sim_dir_name=pkg_resources.resource_filename('libensemble.sim_funcs.branin', '')
from libensemble.sim_funcs.branin.branin_obj import call_branin as obj_func

# Import gen_func
from libensemble.gen_funcs.aposmm import aposmm_logic

script_name = os.path.splitext(os.path.basename(__file__))[0]

### Declare the run parameters/functions
max_sim_budget = 150
n = 2
w = MPI.COMM_WORLD.Get_size()-1

#State the objective function, its arguments, output, and necessary parameters (and their sizes)
sim_specs = {'sim_f': obj_func, # This is the function whose output is being minimized
             'in': ['x'], # These keys will be given to the above function
             'out': [('f',float), # This is the output from the function being minimized
                    ],
             'sim_dir': sim_dir_name, # to be copied by each worker
             'clean_jobs': True,
             }

# As an example, have the workers put their directories in a different
# location. (Useful if a /scratch/ directory is faster than the filesystem.)
# (Otherwise, will just copy in same directory as sim_dir)
if w == 1:
    sim_specs['sim_dir_prefix'] = '~'


if w == 3:
    sim_specs['uniform_random_pause_ub'] = 0.05

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
      ]

# State the generating function, its arguments, output, and necessary parameters.
gen_specs = {'gen_f': aposmm_logic,
             'in': [o[0] for o in gen_out] + ['f', 'returned'],
             'out': gen_out,
             'lb': np.array([-5,0]),
             'ub': np.array([10,15]),
             'initial_sample_size': 20,
             'localopt_method': 'LN_BOBYQA',
             'dist_to_bound_multiple': 0.99,
             'xtol_rel': 1e-3,
             'min_batch_size': w,
             'num_active_gens': 1,
             'batch_mode': True,
             'high_priority_to_best_localopt_runs': True,
             'max_active_runs': 3,
             }

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': max_sim_budget,
                 'elapsed_wallclock_time': 100,
                 'stop_val': ('f', -1), # key must be in sim_specs['out'] or gen_specs['out']
                }

np.random.seed(1)
persis_info = {}
for i in range(MPI.COMM_WORLD.Get_size()):
    persis_info[i] = {'rand_stream': np.random.RandomState(i)}
# Perform the run

if __name__ == "__main__":
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info)

    if MPI.COMM_WORLD.Get_rank() == 0:
        short_name = script_name.split("test_", 1).pop()
        filename = short_name + '_History_length=' + str(len(H)) + '_evals=' + str(sum(H['returned'])) + '_ranks=' + str(w)
        print("\n\n\nRun completed.\nSaving results to file: " + filename)
        np.save(filename, H)

        import pkg_resources; minima_and_func_val_file = pkg_resources.resource_filename('libensemble.sim_funcs.branin', 'known_minima_and_func_values')

        if os.path.isfile(minima_and_func_val_file):
            M = np.loadtxt(minima_and_func_val_file)
            M = M[M[:,-1].argsort()] # Sort by function values (last column)
            k = 3
            tol = 1e-5
            for i in range(k):
                print(np.min(np.sum((H['x'][H['local_min']]-M[i,:n])**2,1)))
                assert np.min(np.sum((H['x'][H['local_min']]-M[i,:n])**2,1)) < tol

            print("\nlibEnsemble with APOSMM has identified the " + str(k) + " best minima within a tolerance " + str(tol))
