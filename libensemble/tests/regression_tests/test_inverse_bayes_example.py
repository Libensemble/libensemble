# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_inverse_bayes_example.py
#    python3 test_inverse_bayes_example.py --nworkers 3 --comms local
#    python3 test_inverse_bayes_example.py --nworkers 3 --comms tcp
#
# Debugging:
#    mpiexec -np 4 xterm -e "python3 inverse_bayes_example.py"
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4

import sys
import numpy as np

from libensemble.libE import libE
from libensemble.sim_funcs.inverse_bayes import likelihood_calculator as sim_f
from libensemble.gen_funcs.persistent_inverse_bayes import persistent_updater_after_likelihood as gen_f
from libensemble.alloc_funcs.inverse_bayes_allocf import only_persistent_gens_for_inverse_bayes as alloc_f
from libensemble.tests.regression_tests.common import parse_args, per_worker_stream

# Parse args for test code
nworkers, is_master, libE_specs, _ = parse_args()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('like', float)]}

gen_specs = {'gen_f': gen_f,
             'in': [],
             'out': [('x', float, 2), ('batch', int), ('subbatch', int),
                     ('prior', float, 1), ('prop', float, 1), ('weight', float, 1)],
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2]),
             'subbatch_size': 3,
             'num_subbatches': 2,
             'num_batches': 10}

persis_info = per_worker_stream({}, nworkers + 1)

# Tell libEnsemble when to stop
exit_criteria = {
    'sim_max': gen_specs['subbatch_size']*gen_specs['num_subbatches']*gen_specs['num_batches'],
    'elapsed_wallclock_time': 300}

alloc_specs = {'out': [], 'alloc_f': alloc_f}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs, libE_specs)

if is_master:
    assert flag == 0
    # Change the last weights to correct values (H is a list on other cores and only array on manager)
    ind = 2*gen_specs['subbatch_size']*gen_specs['num_subbatches']
    H[-ind:] = H['prior'][-ind:] + H['like'][-ind:] - H['prop'][-ind:]
    assert len(H) == 60, "Failed"
