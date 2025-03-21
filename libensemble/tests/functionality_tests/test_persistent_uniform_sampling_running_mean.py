"""
Tests the ability of libEnsemble to
 - give back history entries from the a shutting-down persistent gen

Execute via one of the following commands (e.g., 3 workers):
   mpiexec -np 4 python test_persistent_uniform_sampling_adv.py
   python test_persistent_uniform_sampling_running_mean.py --nworkers 3
   python test_persistent_uniform_sampling_running_mean.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 3 4

import sys

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.persistent_sampling import persistent_uniform_final_update as gen_f
from libensemble.libE import libE
from libensemble.sim_funcs.six_hump_camel import six_hump_camel_simple as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    libE_specs["use_persis_return_gen"] = True

    if nworkers < 2:
        sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

    n = 3
    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "persis_in": ["f", "x", "corner_id", "sim_id"],
        "out": [
            ("sim_id", int),
            ("corner_id", int),
            ("x", float, (n,)),
        ],  # expect ("f_est", float) from gen - test ability for gen to send back "unexpected" field.
        "user": {
            "initial_batch_size": 20,
            "lb": np.array([-3, -2, -1]),
            "ub": np.array([3, 2, 1]),
        },
    }

    alloc_specs = {"alloc_f": alloc_f}

    sim_max = 120
    exit_criteria = {"sim_max": sim_max}
    libE_specs["final_gen_send"] = True
    libE_specs["save_every_k_sims"] = 2

    for run in range(2):
        if run == 2:
            sim_specs["user"] = {
                "rand": True,
                "pause_time": 1e-4,
            }

        persis_info = add_unique_random_streams({}, nworkers + 1)
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            # Check that last saved history agrees with returned history.
            H_saved = np.load(f"libE_history_after_sim_{sim_max}.npy")
            for name in H.dtype.names:
                np.testing.assert_array_equal(H_saved[name], H[name])

            assert np.all(H["f_est"][0:sim_max] != 0), "The persistent gen should have set these at shutdown"
            assert np.all(H["gen_informed"][0:sim_max]), "Need to mark the gen having been informed."
            save_libE_output(H, persis_info, __file__, nworkers)
