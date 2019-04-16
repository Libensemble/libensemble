# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 {FILENAME}.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """
import numpy as np

# Import libEnsemble main, sim_specs, gen_specs, alloc_specs, and persis_info
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt as gen_f
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens as alloc_f
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, give_each_worker_own_stream

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()

if nworkers < 2: # Don't do a "persistent worker run" if only one worker
    quit()

n = 2
sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    'out': [('f', float), ('grad', float, n)]}

gen_out += [('x', float, n), ('x_on_cube', float, n)]
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
    'localopt_method': 'LD_MMA',
    'xtol_rel': 1e-4,}

alloc_specs = {'alloc_f': alloc_f, 'out': gen_out}

persis_info = give_each_worker_own_stream({}, nworkers+1)

exit_criteria = {'sim_max': 1000, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    assert flag == 0

    from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
    tol = 0.1
    for m in minima:
        assert np.min(np.sum((H['x']-m)**2, 1)) < tol

    print("\nlibEnsemble identified the 6 minima to a tolerance "+str(tol))

    save_libE_output(H, __file__, nworkers)
