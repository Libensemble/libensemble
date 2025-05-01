"""
Tests the async-mode of the Tasmanian generator function.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_tasmanian_async.py
   python test_persistent_tasmanian_async.py --nworkers 3
   python test_persistent_tasmanian_async.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local
# TESTSUITE_NPROCS: 4
# TESTSUITE_OS_SKIP: OSX
# TESTSUITE_EXTRA: true

import itertools
import sys
from time import time

import numpy as np

from libensemble.gen_funcs.persistent_tasmanian import get_sparse_grid_specs

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_func
from libensemble.tools import parse_args


# Define some grid initializers.
def tasmanian_init_global():
    # Note: if Tasmanian has been compiled with OpenMP support (i.e., the usual way)
    #       libEnsemble calls cannot be made after the `import Tasmanian` clause
    #       there is a conflict between the OpenMP environment and Python threading
    #       thus Tasmanian has to be imported inside the `tasmanian_init` method
    import Tasmanian

    grid = Tasmanian.makeGlobalGrid(2, 1, 6, "iptotal", "clenshaw-curtis")
    grid.setDomainTransform(np.array([[-5.0, 5.0], [-2.0, 2.0]]))
    return grid


def tasmanian_init_localp():
    import Tasmanian

    grid = Tasmanian.makeLocalPolynomialGrid(2, 1, 3)
    grid.setDomainTransform(np.array([[-5.0, 5.0], [-2.0, 2.0]]))
    return grid


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    # Get node info.
    nworkers, is_manager, libE_specs, _ = parse_args()
    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    # Create an async simulator function (must return 'x' and 'f').
    def sim_f(H, persis_info, sim_specs, _):
        batch = len(H["x"])
        H0 = np.zeros(batch, dtype=sim_specs["out"])
        H0["x"] = H["x"]
        for i, x in enumerate(H["x"]):
            H0["f"][i] = six_hump_camel_func(x)
        return H0, persis_info

    # Set up test parameters.
    user_specs_arr = []
    user_specs_arr.append(
        {
            "refinement": "getCandidateConstructionPoints",
            "tasmanian_init": lambda: tasmanian_init_global(),
            "sType": "iptotal",
            "liAnisotropicWeightsOrOutput": -1,
        }
    )
    user_specs_arr.append(
        {
            "refinement": "getCandidateConstructionPointsSurplus",
            "tasmanian_init": lambda: tasmanian_init_localp(),
            "fTolerance": 1.0e-2,
            "sRefinementType": "classic",
        }
    )
    exit_criteria_arr = []
    exit_criteria_arr.append({"wallclock_max": 3})
    exit_criteria_arr.append({"gen_max": 100})

    run_num = 0
    # Test over all possible parameter combinations.
    for user_specs, exit_criteria in itertools.product(user_specs_arr, exit_criteria_arr):
        sim_specs, gen_specs, alloc_specs, persis_info = get_sparse_grid_specs(user_specs, sim_f, 2, mode="async")

        if run_num == 0:
            gen_specs["user"]["tasmanian_checkpoint_file"] = "tasmanian.grid"
            run_num += 1

        if is_manager:
            print(f"[Manager]: user_specs = {user_specs}")
            print(f"[Manager]: exit_criteria = {exit_criteria}")
            start_time = time()
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)
        if is_manager:
            print("[Manager]: Time taken = ", time() - start_time, "\n", flush=True)
