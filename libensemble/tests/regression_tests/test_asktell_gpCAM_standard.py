"""
Tests libEnsemble with gpCAM

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 3 python test_asktell_gpCAM_standard.py
   python test_asktell_gpCAM_standard.py -n 3

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.

See libensemble.gen_classes.gpCAM for more details about the generator
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
from libensemble.gen_classes.gpCAM import Standard_GP_CAM

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

    vocs = VOCS(variables={"x0": [-3, 3], "x1": [-2, 2], "x2": [-1, 1], "x3": [-1, 1]}, objectives={"f", "MINIMIZE"})
    gen_specs["generator"] = Standard_GP_CAM(vocs, ask_max_iter=1)

    num_batches = 3  # Few because the ask_tell gen can be slow
    exit_criteria = {"sim_max": num_batches * batch_size, "wallclock_max": 300}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert len(np.unique(H["gen_ended_time"])) == num_batches

        save_libE_output(H, persis_info, __file__, nworkers)
