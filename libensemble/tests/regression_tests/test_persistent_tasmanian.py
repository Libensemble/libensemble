# """
# Runs libEnsemble on the 6-hump camel problem. Documented here:
#    https://www.sfu.ca/~ssurjano/camel6.html
#
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_OS_SKIP: OSX

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.persistent_tasmanian import sparse_grid_batched as gen_f_batched
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from time import time

import Tasmanian

nworkers, is_manager, libE_specs, _ = parse_args()

if is_manager:
    start_time = time()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

num_dimensions = 2
sim_specs = {'sim_f': sim_f,
             'in': ['x'],
             'out': [('f', float)]}

gen_specs = {'gen_f': gen_f_batched,
             'in': ['x', 'f'],
             'out': [('x', float, num_dimensions)],
             }

alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

for run in range(2):
    # testing two cases, static construction without refinement
    # and refinement until 100 points have been computed
    persis_info = add_unique_random_streams({}, nworkers + 1)

    # set the stopping criteria
    if run == 0:
        # note that using 'setAnisotropicRefinement' without 'gen_max' will create an infinite loop
        # other stopping criteria could be used with 'setSurplusRefinement' or no refinement
        exit_criteria = {'elapsed_wallclock_time': 10}
    elif run == 1:
        exit_criteria = {'gen_max': 100}  # This will test persistent_tasmanian stopping early.

    # create a sparse grid, will be used only in the persistent generator rank
    grid = Tasmanian.makeGlobalGrid(num_dimensions, 1, 6, "iptotal", "clenshaw-curtis")
    grid.setDomainTransform(np.array([[-5.0, 5.0], [-2.0, 2.0]]))
    gen_specs['user'] = {'grid': grid,
                         'tasmanian_checkpoint_file': 'tasmanian{0}.grid'.format(run)
                         }

    # setup the refinement criteria
    if run == 0:
        gen_specs['user']['refinement'] = 'none'

    if run == 1:
        # See Tasmanian manual: https://ornl.github.io/TASMANIAN/stable/classTasGrid_1_1TasmanianSparseGrid.html
        gen_specs['user']['refinement'] = 'setAnisotropicRefinement'
        gen_specs['user']['sType'] = 'iptotal'
        gen_specs['user']['iMinGrowth'] = 10
        gen_specs['user']['iOutput'] = 0

    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                alloc_specs, libE_specs)

    if is_manager:
        # run sanity check on the computed results
        # this should probably be done on the gen_f rank
        # right now, using the checkpoint file to read the grid from the filesystem
        grid.read(gen_specs['user']['tasmanian_checkpoint_file'])

        if run == 0:
            assert grid.getNumNeeded() == 0, "Correctly left no points needing data"
            assert grid.getNumLoaded() == 49, "Correctly loaded all points"

            print('[Manager]: Time taken =', time() - start_time, flush=True)

            save_libE_output(H, persis_info, __file__, nworkers)

        if run == 1:
            assert grid.getNumNeeded() == 0, "Correctly stopped without completing the refinement"
            assert grid.getNumLoaded() == 89, "Correctly loaded all points"
