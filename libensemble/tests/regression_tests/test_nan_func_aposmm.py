# """
# Runs libEnsemble on a function that returns only nan; tests APOSMM functionality
#
# Execute via the following command:
#    mpiexec -np 4 python3 test_nan_func_aposmm.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import numpy as np

from libensemble.tests.regression_tests.support import save_libE_output

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import nan_func_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import aposmm_without_grad_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import persis_info_0 as persis_info

n = 2
gen_specs['in'] += ['f_i','x','x_on_cube','obj_component']
gen_specs['out'] += [('x',float,n),('x_on_cube',float,n),('obj_component',int)]
gen_specs['lb'] = -2*np.ones(n)
gen_specs['ub'] =  2*np.ones(n)

w = MPI.COMM_WORLD.Get_size()-1
if w == 3:
    gen_specs['single_component_at_a_time'] = True
    gen_specs['components'] = 1
    gen_specs['combine_component_func'] = np.linalg.norm

# Tell libEnsemble when to stop
exit_criteria = {'sim_max': 100, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info)
if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
    assert np.all(~H['local_pt'])

    save_libE_output(H,__file__)
