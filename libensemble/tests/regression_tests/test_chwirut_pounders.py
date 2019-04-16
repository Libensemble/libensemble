# """
# Runs libEnsemble with a simple uniform random sample on one instance of the GKLS
# problem. # Execute via the following command:

# mpiexec -np 4 python3 test_chwirut_pounders.py

# """
import numpy as np

# Import libEnsemble items
from libensemble.libE import libE
from libensemble.sim_funcs.chwirut1 import chwirut_eval as sim_f
from libensemble.gen_funcs.aposmm import aposmm_logic as gen_f
from libensemble.tests.regression_tests.support import persis_info_2 as persis_info, aposmm_gen_out as gen_out, branin_vals_and_minima as M
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, give_each_worker_own_stream

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] == 'local':
    quit()

### Declare the run parameters/functions
m = 214
n = 3
max_sim_budget = 10

sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f',float), ('fvec',float,m),
                     ],
             'combine_component_func': lambda x: np.sum(np.power(x,2)),
             }

gen_out += [('x',float,n), ('x_on_cube',float,n)]

gen_specs = {'gen_f': gen_f,
             'in': [o[0] for o in gen_out] + ['f','fvec', 'returned'],
             'out': gen_out,
             'initial_sample_size': 5,
             'num_active_gens': 1,
             'batch_mode': True,
             'lb': (-2-np.pi/10)*np.ones(n), # Trying to avoid exactly having x[1]=-x[2] from being hit, which results in division by zero in chwirut. 
             'ub':  2*np.ones(n),
             'localopt_method': 'pounders',
             'dist_to_bound_multiple': 0.5,
             'components': m,
             }
gen_specs.update({'grtol': 1e-4, 'gatol': 1e-4, 'frtol': 1e-15, 'fatol': 1e-15})

persis_info = give_each_worker_own_stream(persis_info,nworkers+1)

exit_criteria = {'sim_max': max_sim_budget, 'elapsed_wallclock_time': 300 }

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_master:
    assert flag == 0
    assert len(H) >= max_sim_budget

    # Calculating the Jacobian at the best point (though this information was not used by pounders)
    from libensemble.sim_funcs.chwirut1 import EvaluateJacobian
    J = EvaluateJacobian(H['x'][np.argmin(H['f'])])
    assert np.linalg.norm(J) < 2000

    save_libE_output(H,__file__,nworkers)
