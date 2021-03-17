# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_persistent_uniform_sampling_async.py
#    python3 test_6-hump_camel_persistent_uniform_sampling_async.py --nworkers 3 --comms local
#    python3 test_6-hump_camel_persistent_uniform_sampling_async.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.branin.branin_obj import call_branin as sim_f
from libensemble.gen_funcs.persistent_uniform_sampling import persistent_uniform as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_manager, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float)],
             'user': {'uniform_random_pause_ub': 0.5}
             }

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,))],
             'user': {'gen_batch_size': nworkers - 1,
                      'lb': np.array([-3, -2]),
                      'ub': np.array([3, 2])}
             }

alloc_specs = {'alloc_f': alloc_f,
               'out': [('given_back', bool)],
               'user': {'async_return': True}
               }

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'gen_max': 100, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_manager:
    [_, counts] = np.unique(H['gen_time'], return_counts=True)
    print(counts)
    assert counts[0] == nworkers - 1, "The first gen_time should be common among gen_batch_size number of points"
    assert len(np.unique(counts)) > 1, "There is no variablitiy in the gen_times but there should be for the async case"

    save_libE_output(H, persis_info, __file__, nworkers)
