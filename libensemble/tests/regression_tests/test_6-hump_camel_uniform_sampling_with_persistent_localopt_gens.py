# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_uniform_sampling_with_persistent_localopt_gens.py
#    python3 test_6-hump_camel_uniform_sampling_with_persistent_localopt_gens.py --nworkers 3 --comms local
#    python3 test_6-hump_camel_uniform_sampling_with_persistent_localopt_gens.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

import sys
import numpy as np

# Import libEnsemble main, sim_specs, gen_specs, alloc_specs, and persis_info
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.uniform_or_localopt import uniform_or_localopt as gen_f
from libensemble.alloc_funcs.start_persistent_local_opt_gens import start_persistent_local_opt_gens as alloc_f
from libensemble.tests.regression_tests.support import uniform_or_localopt_gen_out as gen_out
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, per_worker_stream
from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima

nworkers, is_master, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float), ('grad', float, n)]}

gen_out += [('x', float, n), ('x_on_cube', float, n)]
gen_specs = {'gen_f': gen_f,
             'in': [],
             'xtol_rel': 1e-4,
             'out': gen_out,
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2]),
             'gen_batch_size': 2,
             'batch_mode': True,
             'num_active_gens': 1,
             'localopt_method': 'LD_MMA',
             'xtol_rel': 1e-4}

alloc_specs = {'alloc_f': alloc_f, 'out': gen_out}

persis_info = per_worker_stream({}, nworkers + 1)

exit_criteria = {'sim_max': 1000, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    assert flag == 0

    tol = 0.1
    for m in minima:
        assert np.min(np.sum((H['x'] - m)**2, 1)) < tol

    print("\nlibEnsemble found the 6 minima to a tolerance " + str(tol))

    save_libE_output(H, persis_info, __file__, nworkers)
