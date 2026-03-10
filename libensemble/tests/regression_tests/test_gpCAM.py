"""
Tests libEnsemble with gpCAM

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_gpCAM.py
   python test_gpCAM.py --nworkers 3

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.

Runs three variants of gpCAM. The first two use the posterior covariance
sampling method,  whereby the second run uses the grid approach and uses
the points from the first run as it’s test points. The third run uses the
gpCAM ask/tell interface.

See libensemble.gen_funcs.persistent_gpCAM for more details about the
generator setup.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true

import sys
import warnings

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.persistent_gpCAM import persistent_gpCAM, persistent_gpCAM_covar

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

warnings.filterwarnings("ignore", message="Default hyperparameter_bounds")
warnings.filterwarnings("ignore", message="Hyperparameters initialized")

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 4
    batch_size = 15

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [
            ("f", float),
        ],
    }

    gen_specs = {
        "persis_in": ["x", "f", "sim_id"],
        "out": [("x", float, (n,))],
        "user": {
            "batch_size": batch_size,
            "lb": np.array([-3, -2, -1, -1]),
            "ub": np.array([3, 2, 1, 1]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    for inst in range(3):
        if inst == 0:
            gen_specs["gen_f"] = persistent_gpCAM_covar
            num_batches = 10
            exit_criteria = {"sim_max": num_batches * batch_size, "wallclock_max": 300}
            libE_specs["save_every_k_gens"] = 150
            libE_specs["H_file_prefix"] = "gpCAM_nongrid"
        if inst == 1:
            gen_specs["user"]["use_grid"] = True
            gen_specs["user"]["test_points_file"] = "gpCAM_nongrid_after_gen_150.npy"
            libE_specs["final_gen_send"] = True
            del libE_specs["H_file_prefix"]
            del libE_specs["save_every_k_gens"]
        elif inst == 2:
            gen_specs["gen_f"] = persistent_gpCAM
            num_batches = 3  # Few because the ask_tell gen can be slow
            gen_specs["user"]["ask_max_iter"] = 1  # For quicker test
            exit_criteria = {"sim_max": num_batches * batch_size, "wallclock_max": 300}

        persis_info = add_unique_random_streams({}, nworkers + 1)

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            assert len(np.unique(H["gen_ended_time"])) == num_batches

            save_libE_output(H, persis_info, __file__, nworkers)
