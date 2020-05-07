# """
# Runs libEnsemble on a 3-objective problem using the VTMOP package.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_VTMOP.py
#    python3 test_VTMOP.py --nworkers 3 --comms local
#    python3 test_VTMOP.py --nworkers 3 --comms tcp
#
# This requires that VTMOP be installed. Contact thchang@vt.edu for a copy.
# To build, use  : $ make genfuncs
# To add to path : $ export PATH=$PATH:`pwd` (from src/build directory)
#
# The wrapper for VTMOP is in gen_funcs/vtmop.py
# Running this test will generate 3 unformatted binary files mop.io, mop.dat,
# and mop.chkpt in the working directory, for sharing data between VTMOP
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
# from libensemble.sim_funcs.mop_funcs import convex_mop as func
from libensemble.sim_funcs.mop_funcs import dtlz2 as func
from libensemble.gen_funcs.vtmop import vtmop_gen as gen_f
from libensemble.alloc_funcs.vtmop_alloc import give_sim_work_first as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

# Set the problem dimensions here
num_dims = 5
num_objs = 3
lower = 0.0
upper = 1.0


# Definition of the sum function
def sim_f(H, *unused):
    # Initialize the output array
    Out = np.zeros(1, dtype=sim_specs['out'])
    # Evaluate the objective functions
    f = np.ones(np.size(Out['f']))
    func(H['x'][0], f)
    # Return the output array
    Out['f'] = f
    return Out, {}


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
gen_specs = {'gen_f': gen_f,  # Set the generator to VTMOP (aliased to gen_f above).
             'in': ['x', 'f'],
             'out': [('x', float, num_dims)],
             'user': {
                 # Set the number of objectives. The number of design variables is
                 # inferred based on the length of lb.
                 'num_obj': num_objs,
                 # Set the bound constraints.
                 'lb': lower_bounds,
                 'ub': upper_bounds,
                 # search_batch_size is the number of points used to search
                 # each local trust region (using Latin hypercube design).
                 # This should be a multiple of the number of concurrent function
                 # evaluations and on the order of 2*d (where d is the number of
                 # design variables)
                 'search_batch_size': 12,
                 # opt_batch_size is the preferred number of candidate designs.
                 # When the actual number of candidates is not a multiple of
                 # opt_batch_size, additional candidates are randomly generated
                 # to pad out the batch (if possible). This should be the exact
                 # number of concurrent simulations used.
                 'opt_batch_size': 3,
                 # first_batch_size specifies the size of the initial search
                 # and should generally be a large number. However, if a
                 # precomputed database is available, then the initial search
                 # could be skipped. If 0 is given, then the initial search is
                 # skipped. Setting first_batch_size to 0 without supplying an
                 # initial database will cause an error since the surrogates
                 # cannot be fit without sufficient data.
                 'first_batch_size': 1000,
                 # set restart to True, unless you are reloading from a checkpoint
                 'restart': True},
             }

# Set up the allocator
alloc_specs = {'alloc_f': alloc_f, 'out': [('allocated', bool)], 'user': {'num_active_gens': 1}}

# Persistent info between iterations
persis_info = add_unique_random_streams({}, nworkers + 1)
persis_info['next_to_give'] = 0
persis_info['total_gen_calls'] = 0

# Run for 2000 evaluations or 300 seconds
exit_criteria = {'sim_max': 2000, 'elapsed_wallclock_time': 300}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs=alloc_specs, libE_specs=libE_specs)

# The master takes care of checkpointint/output
if is_master:
    assert flag == 0
    save_libE_output(H, persis_info, __file__, nworkers)
