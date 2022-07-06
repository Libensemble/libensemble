"""
Tests the ability of libEnsemble to
 - give back all of the history to a persistent gen at shutdown

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_persistent_uniform_sampling_adv.py
   python test_persistent_uniform_sampling_adv.py --nworkers 3 --comms local
   python test_persistent_uniform_sampling_adv.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 4

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.gen_funcs.persistent_sampling import persistent_uniform as gen_f
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["use_persis_return_gen"] = True

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 2
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float), ("grad", float, n)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "x", "grad", "sim_id"],
        "out": [("x", float, (n,))],
        "user": {
            "initial_batch_size": 100,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    sim_max = 40
    exit_criteria = {"sim_max": sim_max}

    alloc_specs = {"alloc_f": alloc_f, "out": []}

    # Perform the runs
    for prob_id in range(2):
        if prob_id == 0:
            gen_specs["user"]["replace_final_fields"] = False
        else:
            gen_specs["user"]["replace_final_fields"] = True
            libE_specs["final_fields"] = ["x", "f", "sim_id"]

        persis_info = add_unique_random_streams({}, nworkers + 1)

        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            assert len(np.unique(H["gen_ended_time"])) == 1, "Everything should have been generated in one batch"
            if prob_id == 1:
                assert np.all(H["x"][0:sim_max] == -1.23), "The persistent gen should have set these at shutdown"

            save_libE_output(H, persis_info, __file__, nworkers)
