# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command:

# mpiexec -np 4 python3 call_chwirut_aposmm_one_residual_at_a_time.py

# """
import numpy as np

from libensemble.tests.regression_tests.common import parse_args, save_libE_output

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    quit()


# Import libEnsemble main, sim_specs, gen_specs, alloc_specs, and persis_info
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import chwirut_one_at_a_time_sim_specs as sim_specs
from libensemble.tests.regression_tests.support import aposmm_without_grad_gen_specs as gen_specs
from libensemble.tests.regression_tests.support import give_sim_work_first_pausing_alloc_specs as alloc_specs

from libensemble.tests.regression_tests.support import persis_info_3 as persis_info
from libensemble.tests.regression_tests.support import give_each_worker_own_stream 
persis_info = give_each_worker_own_stream(persis_info,nworkers+1)

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 30*m

gen_specs['out'] += [('x',float,n),
                     ('x_on_cube',float,n),
                     ('obj_component',int),
                     ('f',float)]

gen_specs['in'] += ['f_i','x','x_on_cube','obj_component']
gen_specs['lb'] = (-2-np.pi/10)*np.ones(n) # Trying to avoid exactly having x[1]=-x[2] from being hit, which results in division by zero in chwirut. 
gen_specs['ub'] =  2*np.ones(n)
gen_specs['localopt_method'] = 'pounders'
gen_specs['dist_to_bound_multiple'] = 0.5
gen_specs.update({'grtol': 1e-4, 'gatol': 1e-4, 'frtol': 1e-15, 'fatol': 1e-15})
gen_specs['single_component_at_a_time'] = True
gen_specs['components'] = m
gen_specs['combine_component_func'] = lambda x: np.sum(np.power(x,2))

np.random.RandomState(0)
gen_specs['sample_points'] = np.random.uniform(0,1,(max_sim_budget,n))*(gen_specs['ub']-gen_specs['lb'])+gen_specs['lb']

exit_criteria = {'sim_max': max_sim_budget, # must be provided
                 'elapsed_wallclock_time': 300
                  }

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

if is_master:
    assert flag == 0
    assert len(H) >= max_sim_budget

    save_libE_output(H,__file__,nworkers)
