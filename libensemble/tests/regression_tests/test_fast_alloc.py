# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_fast_alloc.py
#    python3 test_fast_alloc.py --nworkers 3 --comms local
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 2 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_simple as sim_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.alloc_funcs.fast_alloc import give_sim_work_first as alloc_f
from libensemble.tests.regression_tests.common import parse_args, per_worker_stream

nworkers, is_master, libE_specs, _ = parse_args()

num_pts = 30*(nworkers - 1)

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'in': ['sim_id'],
             'out': [('x', float, (2,))],
             'gen_batch_size': num_pts,
             'num_active_gens': 1,
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2])}

alloc_specs = {'alloc_f': alloc_f, 'out': [('allocated', bool)]}

persis_info = per_worker_stream({}, nworkers + 1)

exit_criteria = {'sim_max': num_pts, 'elapsed_wallclock_time': 300}

if libE_specs['comms'] == 'tcp':
    # Can't use the same interface for manager and worker if we want
    # repeated calls to libE -- the manager sets up a different server
    # each time, and the worker will not know what port to connect to.
    sys.exit("Cannot run with tcp when repeated calls to libE -- aborting...")

for time in np.append([0], np.logspace(-5, -1, 5)):
    for rep in range(1):
        sim_specs['pause_time'] = time

        if time == 0:
            sim_specs.pop('pause_time')
            gen_specs['gen_batch_size'] = num_pts//2

        persis_info['next_to_give'] = 0
        persis_info['total_gen_calls'] = 1

        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                    persis_info, alloc_specs, libE_specs)

        if is_master:
            assert flag == 0
            assert len(H) == num_pts
