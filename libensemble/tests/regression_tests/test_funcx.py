"""
Runs libEnsemble with Latin hypercube sampling on a simple 1D problem. Tests launching
of sim_f onto remote resource using funcX.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_funcx.py
   python3 test_funcx.py --nworkers 3 --comms local
   python3 test_funcx.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import secrets
import subprocess
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import remote_write_sim_func as sim_f
from libensemble.tests.regression_tests.support import write_uniform_gen_func as gen_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

nworkers, is_manager, libE_specs, _ = parse_args()
libE_specs['safe_mode'] = False

# Configures endpoint for local machine - default settings
subprocess.run(['funcx-endpoint', 'configure', 'test-libe'])
output = subprocess.run(['funcx-endpoint', 'start', 'test-libe'], capture_output=True)
output = output.stderr.decode().split()
endpoint_uuid = output[output.index('uuid:')+1]

sim_specs = {
    'sim_f': sim_f,
    'funcx_endpoint': endpoint_uuid, # endpoint for remote resource on which to run sim_f
    'in': ['x'],
    'out': [('f', float)],
}

gen_specs = {
    'gen_f': gen_f,
    'out': [('x', float, (1,))],
    'user': {
        'gen_batch_size': 19,
        'lb': np.array([-3]),
        'ub': np.array([3]),
    },
}

libE_specs['sim_dirs_make'] = True
libE_specs['ensemble_dir_path'] = './funcx_ensemble_' + secrets.token_hex(nbytes=3)

persis_info = add_unique_random_streams({}, nworkers + 1, seed=1234)

exit_criteria = {'gen_max': 19}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

if is_manager:
    assert len(H) >= 19
    print("\nlibEnsemble with random sampling has generated enough points")
    save_libE_output(H, persis_info, __file__, nworkers)
