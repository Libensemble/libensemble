# """
# Runs libEnsemble on a made up 3-objective problem consisting of
#
#    f1 = six_hump_camel
#    f2 = one_d_func
#    f3 = branin_obj
#
# using the MOP_MOD package for solving multiobjective optimization problems.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_MOP_MOD.py
#    python3 test_MOP_MOD.py --nworkers 3 --comms local
#    python3 test_MOP_MOD.py --nworkers 3 --comms tcp
#
# This requires that MOP_MOD be installed. Contact thchang@vt.edu for a copy.
# To build, use  : $ make genfuncs
# To add to path : $ export PATH=$PATH:`pwd` (from src/build directory)
#
# The wrapper for MOP_MOD is in gen_funcs/mop_mod.py
# Running this test will generate 3 unformatted binary files mop.io, mop.dat,
# and mop.chkpt in the working directory, for sharing data between MOP_MOD
# and libE.
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
from libensemble.gen_funcs.mop_mod import mop_mod_gen as gen_f
from libensemble.alloc_funcs.fast_alloc import give_sim_work_first as alloc_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, per_worker_stream

# Definition of the sum function
def sim_f(H, *unused):
    # Initialize the output array
    O = np.zeros(1, dtype=sim_specs['out'])
    # Evaluate the 3 objective functions
    f1 = func1(H['x'][0])
    f2 = func2(H, {}, {'out': [('f', float)]}, {})[0][0][0]
    f3 = func3(H, {}, {'out': [('f', float)]}, {})[0][0][0]
    # Return the output array
    O['f'] = np.array([f1, f2, f3])
    return O, {}

# Set up the problem
nworkers, is_master, libE_specs, _ = parse_args()

# Set up the simulator
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float, 3)]}

# Set up the generator
gen_specs = {'num_obj': 3,
             'gen_f': gen_f,
             'in': ['x', 'f'],
             'gen_batch_size': 50,
             'num_active_gens': 1,
             'batch_mode': True,
             'out': [('x', float, 2)],
             'lb': np.array([-3.0, -2.0]),
             'ub': np.array([3.0, 2.0])}

# Set up the allocator
alloc_specs = {'alloc_f': alloc_f, 'out': [('allocated', bool)]}

# Persistent info between iterations
persis_info = per_worker_stream({}, nworkers + 1)
persis_info['next_to_give'] = 0
persis_info['total_gen_calls'] = 0

# Run for 500 evaluations or 300 seconds
exit_criteria = {'sim_max': 500, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs=alloc_specs, libE_specs=libE_specs)

# The master takes care of checkpointint/output
if is_master:
    assert flag == 0
    save_libE_output(H, persis_info, __file__, nworkers)
