# """
# Runs libEnsemble with the persistent generator that finds an appropriate
# finite-difference parameter for the sim_f mapping from R^n to R^m around the
# point x.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_persistent_fd_param_finder.py
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.noisy_vector_mapping import func_wrapper as sim_f
from libensemble.gen_funcs.persistent_fd_param_finder import fd_param_finder as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_master, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = 2
m = 3
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float, m)]}

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, (n,))],
             'user': {'x': np.array([1.23, -0.12]),
                      'kmax': 10}
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)]}

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'gen_max': 400}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    assert len(H) < exit_criteria['gen_max'], "Problem didn't stop early, which should have been the case."

    save_libE_output(H, persis_info, __file__, nworkers)
