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
# TESTSUITE_COMMS:
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import secrets

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.tests.regression_tests.support import remote_write_sim_func
from libensemble.tests.regression_tests.support import remote_write_gen_func
from libensemble.tools import parse_args

nworkers, is_manager, libE_specs, _ = parse_args()
libE_specs['safe_mode'] = False

libE_specs['ensemble_dir_path'] = './ensemble_funcx_' + secrets.token_hex(nbytes=3)

sim_specs = {
    'sim_f': remote_write_sim_func,
    'funcx_endpoint': "11111111-1111-1111-1111-111111111111",
    'in': ['x'],
    'out': [('f', "<U10")],
    'user' : {
        'calc_dir': '/home/user/output'
    }
}

gen_specs = {
    'gen_f': remote_write_gen_func,
    'out': [('x', "<U10", (1,))],
}

exit_criteria = {'sim_max': 80}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, libE_specs=libE_specs)
