# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command:

# mpiexec -np 4 python3 test_chwirut_pounders.py

# """

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI # for libE communicator
import numpy as np

from libensemble.tests.regression_tests.support import save_libE_output

# Import libEnsemble main, sim_specs, gen_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import chwirut_all_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import aposmm_without_grad_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import persis_info_2 as persis_info

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 10

gen_specs['out'] += [('x',float,n),
                     ('x_on_cube',float,n),
                     ]

gen_specs['in'] += ['fvec','x','x_on_cube']
gen_specs['lb'] = -2*np.ones(n)
gen_specs['ub'] =  2*np.ones(n)
gen_specs['localopt_method'] = 'pounders'
gen_specs['dist_to_bound_multiple'] = 0.5
gen_specs.update({'grtol': 1e-4, 'gatol': 1e-4, 'frtol': 1e-15, 'fatol': 1e-15})
gen_specs['components'] = m

exit_criteria = {'sim_max': max_sim_budget, # must be provided
                 'elapsed_wallclock_time': 300
                  }

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info)

if MPI.COMM_WORLD.Get_rank() == 0:
    assert flag == 0
    assert len(H) >= max_sim_budget

    # Calculating the Jacobian at the best point (though this information was not used by pounders)
    from libensemble.sim_funcs.chwirut1 import EvaluateJacobian
    J = EvaluateJacobian(H['x'][np.argmin(H['f'])])
    assert np.linalg.norm(J) < 2000

    save_libE_output(H,__file__)
