"""
Runs libEnsemble on a 3-objective problem using the VTMOP package.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_VTMOP.py
   python3 test_VTMOP.py --nworkers 3 --comms local
   python3 test_VTMOP.py --nworkers 3 --comms tcp

This requires that VTMOP be installed. Contact thchang@vt.edu for a copy.
To build, use  : $ make genfuncs
To add to path : $ export PATH=$PATH:`pwd` (from src/build directory)

The wrapper for VTMOP is in gen_funcs/vtmop.py
Running this test will generate 3 unformatted binary files mop.io, mop.dat,
and mop.chkpt in the working directory, for sharing data between VTMOP
and libE.

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_OS_SKIP: OSX
# TESTSUITE_NPROCS: 5
# TESTSUITE_EXTRA: true

import numpy as np
import os
import time
from libensemble.utils.timer import Timer

# Import libEnsemble items for this test
from libensemble.libE import libE

# from libensemble.sim_funcs.mop_funcs import convex_mop as func
from libensemble.sim_funcs.mop_funcs import dtlz2 as func
from libensemble.gen_funcs.vtmop import vtmop_gen as gen_f
from libensemble.alloc_funcs.only_one_gen_alloc import ensure_one_active_gen as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

timer = Timer()

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
nworkers, is_manager, libE_specs, _ = parse_args()
lower_bounds = lower * np.ones(num_dims)
upper_bounds = upper * np.ones(num_dims)

# Set up the simulator
sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    'out': [('f', float, num_objs)],
}

# Set up the generator
gen_specs = {
    'gen_f': gen_f,  # Set the gen to VTMOP (aliased to gen_f above).
    'in': ['x', 'f'],
    'out': [('x', float, num_dims)],
    'user': {
        # Set the problem dimensions (must match lb and ub).
        'd': num_dims,
        'p': num_objs,
        # Set the bound constraints.
        'lb': lower_bounds,
        'ub': upper_bounds,
        # Is this the beginning of a new run?
        'new_run': True,
        # isnb specifies the size of the initial search
        # and should generally be a large number. However, if a
        # precomputed database is available, then the initial search
        # could be skipped. If 0 is given, then the initial search is
        # skipped. Setting first_batch_size to 0 without supplying an
        # initial database will cause an error since the surrogates
        # cannot be fit without sufficient data.
        'isnb': 1000,
        # snb is the number of points used to search
        # each local trust region (using Latin hypercube design).
        # This should be a multiple of the number of concurrent
        # function evaluations and on the order of 4*d (where d is
        # the number of design variables)
        'snb': int(np.ceil(4 * num_dims / nworkers) * nworkers),
        # onb is the preferred number of candidate designs.
        # When the actual number of candidates is not a multiple of
        # opt_batch_size, additional candidates are randomly generated
        # to pad out the batch (if possible). This should be the exact
        # number of concurrent simulations used.
        'onb': nworkers,
        # Other optional arguments below:
        # Set the trust region radius as a fraction of ub[:]-lb[:].
        # This setting is problem dependent. A good starting place
        # would be between 0.1 and 0.2.
        'trust_radf': 0.1,
    },
}

# Set up the allocator
alloc_specs = {'alloc_f': alloc_f, 'out': []}

s1 = []
H = []

for run in range(3):
    if run == 0:
        # Run for 1100 evaluations or 300 seconds
        H0 = None
        exit_criteria = {'sim_max': 1100, 'elapsed_wallclock_time': 300}

    elif run == 1:
        # In the second run, we initialize VTMOP with an initial sample of previously evaluated points
        np.random.seed(0)
        size = 1000

        # Generate the sample
        X = np.random.uniform(gen_specs['user']['lb'], gen_specs['user']['ub'], (size, num_dims))
        f = np.zeros((size, num_objs))

        # Initialize H0
        H0_dtype = [
            ('x', float, num_dims),
            ('f', float, num_objs),
            ('sim_id', int),
            ('returned', bool),
            ('given', bool),
        ]
        H0 = np.zeros(size, dtype=H0_dtype)
        H0['x'] = X
        H0['sim_id'] = range(size)
        H0[['given', 'returned']] = True

        # Perform objective function evaluations
        for i in range(size):
            Out, _ = sim_f(H0[[i]])
            H0['f'][i] = Out['f']

        # Run for 200 more evaluations or 300 seconds
        exit_criteria = {'sim_max': 200, 'elapsed_wallclock_time': 300}

        gen_specs['user']['isnb'] = 0
        gen_specs['user']['new_run'] = True  # Need to set this as it can be overwritten within the libE call.

    elif run == 2:
        # In the third run, we restart VTMOP by loading in the history array saved in run==1
        gen_specs['user']['new_run'] = False

        # Inelegant way to have the manager copy over the VTMOP checkpoint
        # file, and have every worker get the H value from the run==1 case to
        # use in the restart.
        try:
            os.remove('manager_done_file')
        except OSError:
            pass

        if is_manager:
            os.rename('vtmop.chkpt_finishing_' + s1, 'vtmop.chkpt')
            np.save('H_for_vtmop_restart.npy', H)
            open('manager_done_file', 'w').close()
        else:
            while not os.path.isfile('manager_done_file'):
                time.sleep(0.1)
            H = np.load('H_for_vtmop_restart.npy')

        # Initialize H0 with values from H (from the run==1 case)
        size = sum(H['returned'])
        H0_dtype = [
            ('x', float, num_dims),
            ('f', float, num_objs),
            ('sim_id', int),
            ('returned', bool),
            ('given', bool),
        ]
        H0 = np.zeros(size, dtype=H0_dtype)
        H0['x'] = H['x'][:size]
        H0['sim_id'] = range(size)
        H0[['given', 'returned']] = True
        H0['f'] = H['f'][:size]

        # Run for 200 more evaluations or 300 seconds
        exit_criteria = {'sim_max': 200, 'elapsed_wallclock_time': 300}

    # Persistent info between iterations
    persis_info = add_unique_random_streams({}, nworkers + 1)
    persis_info['next_to_give'] = 0 if H0 is None else len(H0)
    persis_info['total_gen_calls'] = 0

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs=alloc_specs, libE_specs=libE_specs, H0=H0
    )

    # The manager takes care of checkpointing/output
    if is_manager:
        # Renaming vtmop checkpointing file, if needed for later use.
        timer.start()
        s1 = timer.date_start.replace(' ', '_')
        os.rename('vtmop.chkpt', 'vtmop.chkpt_finishing_' + s1)

        assert flag == 0
        save_libE_output(H, persis_info, __file__, nworkers)
