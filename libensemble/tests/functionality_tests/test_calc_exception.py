"""
A test of libEnsemble exception handling.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_calc_exception.py
   python test_calc_exception.py --nworkers 3
   python test_calc_exception.py --nworkers 3 --comms tcp
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.libE import libE
from libensemble.manager import LoggedException
from libensemble.tools import add_unique_random_streams, parse_args


# Define sim_func
def six_hump_camel_err(H, persis_info, sim_specs, _):
    raise Exception("Deliberate error")


# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()

    sim_specs = {
        "sim_f": six_hump_camel_err,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, 2)],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "gen_batch_size": 10,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"wallclock_max": 10}

    libE_specs["abort_on_exception"] = False

    # Perform the run
    return_flag = 1
    try:
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)
    except LoggedException as e:
        print(f"Caught deliberate exception: {e}")
        return_flag = 0

    if is_manager:
        assert return_flag == 0
