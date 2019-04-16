# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via the following command:
#    mpiexec -np 4 python3 {FILENAME}.py
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, give_each_worker_own_stream

nworkers, is_master, libE_specs, _ = parse_args()

if nworkers < 2:
    quit()

n = 2
sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    'out': [('f', float), ('grad', float, n)]}

gen_specs = {
    'gen_f': gen_f,
    'in': [],
    'gen_batch_size': 20,
    'out': [('x', float, (n,))],
    'lb': np.array([-3, -2]),
    'ub': np.array([3, 2]),}

alloc_specs = {'alloc_f': alloc_f, 'out': []}

persis_info = give_each_worker_own_stream({}, nworkers+1)

exit_criteria = {'sim_max': 40, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    assert len(np.unique(H['gen_time'])) == 2

    save_libE_output(H, __file__, nworkers)
