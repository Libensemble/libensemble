"""
Runs libEnsemble on the 6-hump camel problem. Documented here:
   https://www.sfu.ca/~ssurjano/camel6.html

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_sim_uniform_sampling.py
   python test_persistent_sim_uniform_sampling.py --nworkers 3
   python test_persistent_sim_uniform_sampling.py --nworkers 3 --comms tcp

When running with the above command, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 3 4
# TESTSUITE_OS_SKIP: WIN

import sys

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import persistent_six_hump_camel as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# from libensemble import logger
# logger.set_level("DEBUG")

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["num_resource_sets"] = nworkers - 1  # Only matters if sims use resources.

    # Only used to test returning/overwriting a point at the end of the persistent sim.
    libE_specs["use_persis_return_sim"] = True

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "user": {"replace_final_fields": True},
        "out": [("f", float), ("grad", float, n)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": [],
        "persis_in": ["sim_id", "f", "grad"],
        "out": [("x", float, (n,))],
        "user": {
            "initial_batch_size": 5,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f, "user": {"alt_type": True}}

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 40, "wallclock_max": 300}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    if is_manager:
        assert len(np.unique(H["gen_ended_time"])) == 8
        assert not any((H["f"] == 0))
        # Should overwrite the last value (in fact last (nworker-1) values) with f(1,1) = 3.23333333
        assert not np.isclose(H["f"][0], 3.23333333)
        assert np.isclose(H["f"][-1], 3.23333333)
        save_libE_output(H, persis_info, __file__, nworkers)
