# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command:

# mpiexec -np 4 python3 test_chwirut_uniform_sampling_one_residual_at_a_time.py

# """

from __future__ import division
from __future__ import absolute_import

import numpy as np
import copy

from libensemble.tests.regression_tests.support import save_libE_output
from libensemble.tests.regression_tests.common import parse_args

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    quit()

# Import libEnsemble main, sim_specs, gen_specs, alloc_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import chwirut_one_at_a_time_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import uniform_random_sample_obj_components_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import give_sim_work_first_pausing_alloc_specs as alloc_specs
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info

from libensemble.tests.regression_tests.support import give_each_worker_own_stream 
persis_info = give_each_worker_own_stream(persis_info,nworkers+1)
persis_info_safe = copy.deepcopy(persis_info)

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 10*m

sim_specs['component_nan_frequency'] = 0.01

gen_specs['out'] += [('x',float,n),]
gen_specs['lb'] = (-2-np.pi/10)*np.ones(n) # Trying to avoid exactly having x[1]=-x[2] from being hit, which results in division by zero in chwirut. 
gen_specs['ub'] =  2*np.ones(n)
gen_specs['components'] = m

exit_criteria = {'sim_max': max_sim_budget, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
if is_master:
    assert flag == 0

    save_libE_output(H,__file__,nworkers)

# Perform the run but not stopping on NaNs
alloc_specs.pop('stop_on_NaNs')
persis_info = copy.deepcopy(persis_info_safe) 
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
if is_master:
    assert flag == 0

# Perform the run also not stopping on partial fvec evals
alloc_specs.pop('stop_partial_fvec_eval')
persis_info = copy.deepcopy(persis_info_safe) 
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
if is_master:
    assert flag == 0
