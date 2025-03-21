"""
Runs libEnsemble to test basic worker/manager communications
Scale up array_size and number of workers as required

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_comms.py
   python test_comms.py --nworkers 3
   python test_comms.py --nworkers 3 --comms tcp

The number of concurrent evaluations of the objective function will be N-1.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import numpy as np

from libensemble.executors.mpi_executor import MPIExecutor  # Only used to get workerID in float_x1000
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.comms_testing import float_x1000 as sim_f
from libensemble.tools import add_unique_random_streams, parse_args, save_libE_output

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":
    nworkers, is_manager, libE_specs, _ = parse_args()
    libE_specs["disable_resource_manager"] = True
    exctr = MPIExecutor()

    array_size = int(1e6)  # Size of large array in sim_specs
    rounds = 2  # Number of work units for each worker
    sim_max = nworkers * rounds

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("arr_vals", float, array_size), ("scal_val", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, (2,))],
        "user": {
            "lb": np.array([-3, -2]),
            "ub": np.array([3, 2]),
            "gen_batch_size": sim_max,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"sim_max": sim_max, "wallclock_max": 300}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        assert flag == 0
        for i in range(sim_max):
            x1 = H["x"][i][0] * 1000.0
            x2 = H["x"][i][1]
            assert np.all(H["arr_vals"][i] == x1), "Array values do not all match"
            assert H["scal_val"][i] == x2 + x2 / 1e7, "Scalar values do not all match"

        save_libE_output(H, persis_info, __file__, nworkers)
