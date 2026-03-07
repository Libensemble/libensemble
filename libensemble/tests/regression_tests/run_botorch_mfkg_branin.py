"""
Example of multi-fidelity optimization using a persistent BoTorch MFKG gen_func.

This test uses the gen_on_manager option (persistent generator runs on
a thread). Therefore nworkers is the number of simulation workers.

Execute via one of the following commands:
   mpiexec -np 5 python run_botorch_mfkg_branin.py
   python run_botorch_mfkg_branin.py --nworkers 4
   python run_botorch_mfkg_branin.py --nworkers 4 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 3.

"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true
# TESTSUITE_OS_SKIP: OSX

import numpy as np

from libensemble import logger
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.gen_funcs.persistent_botorch_mfkg_branin import persistent_botorch_mfkg
from libensemble.libE import libE
from libensemble.sim_funcs.augmented_branin import augmented_branin
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["gen_on_manager"] = True

    sim_specs = {
        "sim_f": augmented_branin,
        "in": ["x", "fidelity"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": persistent_botorch_mfkg,
        # "in": ["sim_id", "x", "f", "fidelity"],
        "persis_in": ["sim_id", "x", "f", "fidelity"],
        "out": [
            ("x", float, (2,)),
            ("fidelity", float),
        ],
        "user": {
            "lb": np.array([0.0, 0.0]),
            "ub": np.array([1.0, 1.0]),
            "n_init_samples": 4,  # Each of these points will have a high-fidelity and low-fidelity evaluation 
            "q": 2,
        },
    }

    alloc_specs = {
        "alloc_f": only_persistent_gens,
        "user": {"async_return": False},
    }

    # libE logger
    logger.set_level("INFO")

    # Exit criteria
    exit_criteria = {"sim_max": 12}  # Exit after running sim_max simulations

    # Create a different random number stream for each worker and the manager
    persis_info = add_unique_random_streams({}, nworkers + 1)

    # Run LibEnsemble, and store results in history array H
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    # Save results to numpy file
    if is_manager:
        save_libE_output(H, persis_info, __file__, nworkers)

