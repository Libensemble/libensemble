# """
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import numpy as np

from libensemble.tests.regression_tests.support import save_libE_output

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import branin_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import aposmm_without_grad_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import persis_info_2 as persis_info


### Declare the run parameters/functions
w = MPI.COMM_WORLD.Get_size()-1

# As an example, have the workers put their directories in a different
# location. (Useful if a /scratch/ directory is faster than the filesystem.)
# (Otherwise, will just copy in same directory as sim_dir)
if w == 1:
    sim_specs['sim_dir_prefix'] = '~'

if w == 3:
    sim_specs['uniform_random_pause_ub'] = 0.05

n=2
gen_specs['in'] += ['x','x_on_cube']
gen_specs['out'] += [('x',float,n), ('x_on_cube',float,n),]
gen_specs['lb'] = np.array([-5,0])
gen_specs['ub'] = np.array([10,15])
gen_specs['initial_sample_size'] = 20
gen_specs['localopt_method'] = 'LN_BOBYQA'
gen_specs['dist_to_bound_multiple'] = 0.99
gen_specs['xtol_rel'] = 1e-3
gen_specs['min_batch_size'] = w
gen_specs['high_priority_to_best_localopt_runs'] = True
gen_specs['max_active_runs'] = 3

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 150,
                 'elapsed_wallclock_time': 100,
                 'stop_val': ('f', -1), # key must be in sim_specs['out'] or gen_specs['out']
                }


# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info)

if MPI.COMM_WORLD.Get_rank() == 0:
    save_libE_output(H,__file__)

    from libensemble.tests.regression_tests.support import branin_vals_and_minima as M

    M = M[M[:,-1].argsort()] # Sort by function values (last column)
    k = 3; tol = 1e-5
    for i in range(k):
        print(np.min(np.sum((H['x'][H['local_min']]-M[i,:2])**2,1)))
        assert np.min(np.sum((H['x'][H['local_min']]-M[i,:2])**2,1)) < tol

    print("\nlibEnsemble with APOSMM has identified the " + str(k) + " best minima within a tolerance " + str(tol))
