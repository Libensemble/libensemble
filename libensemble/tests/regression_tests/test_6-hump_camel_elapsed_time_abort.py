# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_elapsed_time_abort.py
#    python3 test_6-hump_camel_elapsed_time_abort.py --nworkers 3 --comms local
#    python3 test_6-hump_camel_elapsed_time_abort.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.uniform_sampling import uniform_random_sample as gen_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, per_worker_stream, eprint

nworkers, is_master, libE_specs, _ = parse_args()

sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float)],
             'pause_time': 2}

gen_specs = {'gen_f': gen_f,
             'in': ['sim_id'],
             'gen_batch_size': 5,
             'num_active_gens': 1,
             'batch_mode': False,
             'out': [('x', float, (2,))],
             'lb': np.array([-3, -2]),
             'ub': np.array([3, 2])}

persis_info = per_worker_stream({}, nworkers + 1)

exit_criteria = {'elapsed_wallclock_time': 1}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:
    eprint(flag)
    eprint(H)
    assert flag == 2
    save_libE_output(H, persis_info, __file__, nworkers)
