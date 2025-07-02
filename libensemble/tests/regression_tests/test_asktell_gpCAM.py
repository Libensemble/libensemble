"""
Tests libEnsemble with gpCAM

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_gpCAM_class.py
   python test_gpCAM_class.py --nworkers 3 --comms local

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.

See libensemble.gen_funcs.persistent_gpCAM for more details about the generator
setup.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 4
# TESTSUITE_EXTRA: true
# TESTSUITE_EXCLUDE: true

import sys
import warnings

import numpy as np
from generator_standard.vocs import VOCS

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_classes.gpCAM import GP_CAM, GP_CAM_Covar, Standard_GP_CAM

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.rosenbrock import rosenbrock_eval as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

warnings.filterwarnings("ignore", message="Default hyperparameter_bounds")


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

    persis_info = add_unique_random_streams({}, nworkers + 1)

    gen = GP_CAM_Covar(None, persis_info[1], gen_specs, None)

    for inst in range(4):
        if inst == 0:
            gen_specs["generator"] = gen
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
            persis_info = add_unique_random_streams({}, nworkers + 1)
            gen_specs["generator"] = GP_CAM(None, persis_info[1], gen_specs, None)
            num_batches = 3  # Few because the ask_tell gen can be slow
            gen_specs["user"]["ask_max_iter"] = 1  # For quicker test
            exit_criteria = {"sim_max": num_batches * batch_size, "wallclock_max": 300}
        elif inst == 3:
            vocs = VOCS(
                variables={"x0": [-3, 3], "x1": [-2, 2], "x2": [-1, 1], "x3": [-1, 1]}, objectives={"f", "MINIMIZE"}
            )
            gen_specs["generator"] = Standard_GP_CAM(vocs, ask_max_iter=1)
            num_batches = 3  # Few because the ask_tell gen can be slow
            exit_criteria = {"sim_max": num_batches * batch_size, "wallclock_max": 300}

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            assert len(np.unique(H["gen_ended_time"])) == num_batches

            save_libE_output(H, persis_info, __file__, nworkers)
