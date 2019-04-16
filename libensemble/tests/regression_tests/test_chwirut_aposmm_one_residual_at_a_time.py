# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command:

# mpiexec -np 4 python3 call_chwirut_aposmm_one_residual_at_a_time.py

# """
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f
from libensemble.gen_funcs.aposmm import aposmm_logic as gen_f
from libensemble.alloc_funcs.fast_alloc_and_pausing import give_sim_work_first as alloc_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, give_each_worker_own_stream
from libensemble.tests.regression_tests.support import persis_info_3 as persis_info, aposmm_gen_out as gen_out, branin_vals_and_minima as M

nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    quit()

### Declare the run parameters/functions
m = 214
n = 3
budget = 30*m

sim_specs = {
    'sim_f': sim_f,
    'in': ['x', 'obj_component'],
    'out': [('f_i', float)],}

gen_out += [('x', float, n), ('x_on_cube', float, n), ('obj_component', int),
            ('f', float)]

# LB tries to avoid x[1]=-x[2], which results in division by zero in chwirut.
UB = 2*np.ones(n),
LB = (-2-np.pi/10)*np.ones(n),

gen_specs = {
    'gen_f': gen_f,
    'in': [o[0] for o in gen_out]+['f_i', 'returned'],
    'out': gen_out,
    'initial_sample_size': 5,
    'num_active_gens': 1,
    'batch_mode': True,
    'lb': LB,
    'ub': UB,
    'localopt_method': 'pounders',
    'dist_to_bound_multiple': 0.5,
    'single_component_at_a_time': True,
    'components': m,
    'combine_component_func': lambda x: np.sum(np.power(x, 2)),}
gen_specs.update({
    'grtol': 1e-4,
    'gatol': 1e-4,
    'frtol': 1e-15,
    'fatol': 1e-15})
np.random.RandomState(0)
gen_specs['sample_points'] = np.random.uniform(0, 1, (budget, n))*(UB-LB)+LB
alloc_specs = {
    'alloc_f': alloc_f,
    'out': [('allocated', bool)],
    'stop_on_NaNs': True,
    'stop_partial_fvec_eval': True,}

persis_info = give_each_worker_own_stream(persis_info, nworkers+1)

exit_criteria = {
    'sim_max': budget, # must be provided
    'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    assert flag == 0
    assert len(H) >= budget

    save_libE_output(H, __file__, nworkers)
