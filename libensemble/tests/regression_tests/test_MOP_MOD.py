# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_uniform_sampling.py
#    python3 test_6-hump_camel_uniform_sampling.py --nworkers 3 --comms local
#    python3 test_6-hump_camel_uniform_sampling.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func as func1
from libensemble.sim_funcs.one_d_func import one_d_example as func2
from libensemble.sim_funcs.branin.branin_obj import call_branin as func3
from libensemble.gen_funcs.mop_mod import mop_mod_wrapper as gen_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, per_worker_stream


def sim_f(H, *unused):

    O = np.zeros(1, dtype=sim_specs['out'])

    f1 = func1(H['x'][0])
    f2 = func2(H, {}, {'out': [('f', float)]}, {})[0][0][0]
    f3 = func3(H, {}, {'out': [('f', float)]}, {})[0][0][0]

    O['f'] = np.array([f1, f2, f3])

    return O, {}


nworkers, is_master, libE_specs, _ = parse_args()

sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float, 3)]}

gen_specs = {'gen_f': gen_f,
             'in': [],
             'gen_batch_size': 500,
             'out': [('x', float, (2,))],
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2])}

persis_info = per_worker_stream({}, nworkers + 1)

exit_criteria = {'sim_max': 1000, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:
    assert flag == 0
    save_libE_output(H, persis_info, __file__, nworkers)
