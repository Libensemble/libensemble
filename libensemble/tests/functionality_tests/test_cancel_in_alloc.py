"""
Runs libEnsemble in order to test the ability of an allocation function to
cancel long-running simulations. In this case, the simulation has a run-time
in seconds that is drawn uniformly from [0,10] and any time the allocation
function is called and a sim_id has been evaluated for more than 5 seconds,
it is cancelled.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_cancel_in_alloc.py
   python test_cancel_in_alloc.py --nworkers 3
   python test_cancel_in_alloc.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

from libensemble.alloc_funcs.give_sim_work_first import give_sim_work_first
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.branin.branin_obj import call_branin as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["kill_canceled_sims"] = True

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
        "user": {"uniform_random_pause_ub": 10},
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, (2,))],
        "user": {
            "gen_batch_size": 5,
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
        },
    }

    alloc_specs = {
        "alloc_f": give_sim_work_first,
        "user": {
            "cancel_sims_time": 3,
            "batch_mode": False,
            "num_active_gens": 1,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": 10, "wallclock_max": 300}

    # Perform the run
    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs, alloc_specs=alloc_specs
    )

    if is_manager:
        test = np.any(H["cancel_requested"]) and np.any(H["kill_sent"])
        assert test, "This test should have requested a cancellation and had a kill sent"
        save_libE_output(H, persis_info, __file__, nworkers)
