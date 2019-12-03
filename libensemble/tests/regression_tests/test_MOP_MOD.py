# """
# Runs libEnsemble on a 3-objective problem using the MOP_MOD package.
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

# Set the problem dimensions here
num_dims = 5
num_objs = 3
lower = 0.0
upper = 1.0

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.mop_funcs import dtlz2 as func
#from libensemble.sim_funcs.mop_funcs import conv_mop as func
from libensemble.gen_funcs.mop_mod import mop_mod_gen as gen_f
from libensemble.alloc_funcs.fast_alloc import give_sim_work_first as alloc_f
from libensemble.tests.regression_tests.common import parse_args, save_libE_output, per_worker_stream

# Definition of the sum function
def sim_f(H, *unused):
    # Initialize the output array
    O = np.zeros(1, dtype=sim_specs['out'])
    # Evaluate the objective functions
    f = np.ones(np.size(O['f']))
    func(H['x'][0],f)
    # Return the output array
    O['f'] = f
    return O, {}

# Set up the problem
nworkers, is_master, libE_specs, _ = parse_args()
lower_bounds = np.zeros(num_dims)
lower_bounds[:] = lower
upper_bounds = np.ones(num_dims)
upper_bounds[:] = upper

# Set up the simulator
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float, num_objs)]}

# Set up the generator
gen_specs = {'num_obj': num_objs,
             'gen_f': gen_f,
             'in': ['x', 'f'],
             'gen_batch_size': 36,
             'first_batch_size': 900,
             'num_active_gens': 1,
             'batch_mode': True,
             'out': [('x', float, num_dims)],
             'lb': lower_bounds,
             'ub': upper_bounds}

# Set up the allocator
alloc_specs = {'alloc_f': alloc_f, 'out': [('allocated', bool)]}

# Persistent info between iterations
persis_info = per_worker_stream({}, nworkers + 1)
persis_info['next_to_give'] = 0
persis_info['total_gen_calls'] = 0

# Run for 500 evaluations or 300 seconds
exit_criteria = {'sim_max': 2000, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs=alloc_specs, libE_specs=libE_specs)

# The master takes care of checkpointint/output
if is_master:
    assert flag == 0
    save_libE_output(H, persis_info, __file__, nworkers)
