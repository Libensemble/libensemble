# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 {FILENAME}.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

import numpy as np

# Import libEnsemble requirements
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt as gen_f
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens as alloc_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, give_each_worker_own_stream
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_out += [('x', float, 2), ('x_on_cube', float, 2)]
gen_specs = {
    'gen_f': gen_f,
    'in': [],
    'localopt_method': 'LN_BOBYQA',
    'xtol_rel': 1e-4,
    'out': gen_out,
    'lb': np.array([-3, -2]),
    'ub': np.array([3, 2]),
    'gen_batch_size': 2,
    'batch_mode': True,
    'num_active_gens': 1,
    'dist_to_bound_multiple': 0.5,
    'localopt_maxeval': 4,}

alloc_specs = {
    'alloc_f': alloc_f,
    'out': gen_out,}

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()

persis_info = give_each_worker_own_stream({}, nworkers+1)

# Tell libEnsemble when to stop
exit_criteria = {
    'sim_max': 10,
    'elapsed_wallclock_time': 300
} # Intentially set low so as to test that a worker in persistent mode can be terminated correctly

if nworkers < 2:
    # Can't do a "persistent worker run" if only one worker
    quit()

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    assert flag == 0
    save_libE_output(H, __file__, nworkers)
