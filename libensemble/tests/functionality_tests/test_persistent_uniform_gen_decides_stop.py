"""
Test the libEnsemble capability to honor a generator function's request to
stop a run.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 5 python test_persistent_uniform_gen_decides_stop.py
   python test_persistent_uniform_gen_decides_stop.py --nworkers 4
   python test_persistent_uniform_gen_decides_stop.py --nworkers 4 --comms tcp

The number of concurrent evaluations of the objective function with 2 gens will be 2:
5 - 1 manager - 2 persistent gens = 2.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local
# TESTSUITE_NPROCS: 5 7
# TESTSUITE_OS_SKIP: WIN

import sys

import numpy as np

from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.gen_funcs.persistent_sampling import persistent_request_shutdown as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.branin.branin_obj import call_branin as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    for ngens in range(1, 3):
        n = 2
        init_batch_size = nworkers - ngens

        if ngens >= nworkers:
            sys.exit("The number of generators must be less than the number of workers -- aborting...")

        sim_specs = {
            "sim_f": sim_f,
            "in": ["x"],
            "out": [("f", float)],
            "user": {"uniform_random_pause_ub": 0.5},
        }

        gen_specs = {
            "gen_f": gen_f,
            "persis_in": ["f", "x", "sim_id"],
            "out": [("x", float, (n,))],
            "user": {
                "initial_batch_size": init_batch_size,
                "shutdown_limit": 10,  # Iterations on a gen before it triggers a shutdown.
                "lb": np.array([-3, -2]),
                "ub": np.array([3, 2]),
            },
        }

        alloc_specs = {
            "alloc_f": alloc_f,
            "user": {
                "async_return": True,
                "num_active_gens": ngens,
            },
        }

        persis_info = add_unique_random_streams({}, nworkers + 1)

        exit_criteria = {"gen_max": 50, "wallclock_max": 300}

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

        if is_manager:
            [ended_times, counts] = np.unique(H["gen_ended_time"], return_counts=True)
            print("Num. points in each gen iteration:", counts)
            assert (
                counts[0] == init_batch_size
            ), "The first gen_ended_time should be common among initial_batch_size number of points"
            assert (
                sum(counts == init_batch_size) >= ngens
            ), "The initial batch of each gen should be common among initial_batch_size number of points"
            assert len(counts) > 1, "All gen_ended_times are the same; they should be different for the async case"

            gen_workers = np.unique(H["gen_worker"])
            print("Generators that issued points", gen_workers)
            assert (
                len(gen_workers) == ngens
            ), f"The number of gens used {len(gen_workers)} does not match num_active_gens {ngens}"

            save_libE_output(H, persis_info, __file__, nworkers)
