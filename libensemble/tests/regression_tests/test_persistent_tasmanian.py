"""
Tests the batch-mode of the Tasmanian generator function.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_tasmanian.py
   python test_persistent_tasmanian.py --nworkers 3
   python test_persistent_tasmanian.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_OS_SKIP: OSX
# TESTSUITE_EXTRA: true

import sys
from time import time

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.persistent_tasmanian import sparse_grid_batched as gen_f_batched

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output


def tasmanian_init_global():
    # Note: if Tasmanian has been compiled with OpenMP support (i.e., the usual way)
    #       libEnsemble calls cannot be made after the `import Tasmanian` clause
    #       there is a conflict between the OpenMP environment and Python threading
    #       thus Tasmanian has to be imported inside the `tasmanian_init` method
    import Tasmanian

    grid = Tasmanian.makeGlobalGrid(num_dimensions, 1, 6, "iptotal", "clenshaw-curtis")
    grid.setDomainTransform(np.array([[-5.0, 5.0], [-2.0, 2.0]]))
    return grid


def tasmanian_init_localp():
    import Tasmanian

    grid = Tasmanian.makeLocalPolynomialGrid(num_dimensions, 1, 3)
    grid.setDomainTransform(np.array([[-5.0, 5.0], [-2.0, 2.0]]))
    return grid


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    if is_manager:
        start_time = time()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    num_dimensions = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f_batched,
        "persis_in": ["x", "f", "sim_id"],
        "out": [("x", float, num_dimensions)],
    }

    alloc_specs = {"alloc_f": alloc_f}

    grid_files = []

    for run in range(3):
        # testing two cases, static construction without refinement
        # and refinement until 100 points have been computed
        persis_info = add_unique_random_streams({}, nworkers + 1)

        # set the stopping criteria
        if run != 1:
            # note that using 'setAnisotropicRefinement' without 'gen_max' will create an infinite loop
            # other stopping criteria could be used with 'setSurplusRefinement' or no refinement
            exit_criteria = {"wallclock_max": 10}
        elif run == 1:
            exit_criteria = {"gen_max": 100}  # This will test persistent_tasmanian stopping early.

        # tasmanian_init has to be a method that returns an initialized TasmanianSparseGrid object
        # tasmanian_checkpoint_file will be overwritten between each step of the iterative refinement
        #   the final grid will also be stored in the file
        gen_specs["user"] = {
            "tasmanian_init": tasmanian_init_global if run < 2 else tasmanian_init_localp,
            "tasmanian_checkpoint_file": f"tasmanian{run}.grid",
        }

        # setup the refinement criteria
        if run == 0:
            gen_specs["user"]["refinement"] = "none"

        if run == 1:
            # See Tasmanian manual: https://ornl.github.io/TASMANIAN/stable/classTasGrid_1_1TasmanianSparseGrid.html
            gen_specs["user"]["refinement"] = "setAnisotropicRefinement"
            gen_specs["user"]["sType"] = "iptotal"
            gen_specs["user"]["iMinGrowth"] = 10
            gen_specs["user"]["iOutput"] = 0

        if run == 2:
            gen_specs["user"]["refinement"] = "setSurplusRefinement"
            gen_specs["user"]["fTolerance"] = 1.0e-2
            gen_specs["user"]["sCriteria"] = "classic"
            gen_specs["user"]["iOutput"] = 0

        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            grid_files.append(gen_specs["user"]["tasmanian_checkpoint_file"])

            if run == 0:
                print("[Manager]: Time taken =", time() - start_time, flush=True)

                save_libE_output(H, persis_info, __file__, nworkers)

    if is_manager:
        # run sanity check on the computed results
        # using the checkpoint file to read the grids from the filesystem
        # Note: cannot make any more libEnsemble calls after importing Tasmanian,
        #       see the earlier note in tasmanian_init_global()
        import Tasmanian

        assert len(grid_files) == 3, "Failed to generate three Tasmanian grid files"

        for run in range(len(grid_files)):
            grid = Tasmanian.SparseGrid()
            grid.read(grid_files[run])
            if run == 0:
                assert grid.getNumNeeded() == 0, "Failed to leave no points needing data"
                assert grid.getNumLoaded() == 49, "Failed to load all points"

            if run == 1:
                assert grid.getNumNeeded() == 0, "Failed to stop after completing the refinement iteration"
                assert grid.getNumLoaded() == 89, "Failed to load all points"

            if run == 2:
                assert grid.getNumNeeded() == 0, "Failed to stop after completing the refinement iteration"
                assert grid.getNumLoaded() == 93, "Failed to load all points"
