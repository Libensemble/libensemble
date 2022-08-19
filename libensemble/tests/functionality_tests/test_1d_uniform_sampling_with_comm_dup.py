"""
Runs libEnsemble with uniform random sampling on a simple 1D problem
without specifying the default communicator. Tests that mpi is taken
as default, with a duplicate of MPI.COMM_WORLD. If libEnsemble
uses MPI.COMM_WORLD, this test will fail.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_1d_uniform_sampling_with_comm_dup.py

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4
# TESTSUITE_OS_SKIP: WIN

import sys
import numpy as np

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.sim_funcs.one_d_func import one_d_example as sim_f
from libensemble.gen_funcs.sampling import uniform_random_sample as gen_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    if libE_specs["comms"] != "mpi":
        sys.exit("This test only runs with MPI -- aborting...")
        # After this check will remove libE_specs['comms']
    else:
        from mpi4py import MPI

    libE_specs = None  # Let MPI use defaults

    # Check independence of default communicator from MPI.COMM_WORLD
    world = MPI.COMM_WORLD
    if not is_manager:
        world.isend(world.Get_rank(), dest=0, tag=0)

    sim_specs = {
        "sim_f": sim_f,
        "in": ["x"],
        "out": [("f", float)],
    }

    gen_specs = {
        "gen_f": gen_f,
        "in": ["sim_id"],
        "out": [("x", float, (1,))],
        "user": {
            "lb": np.array([-3]),
            "ub": np.array([3]),
            "gen_batch_size": 500,
        },
    }

    persis_info = add_unique_random_streams({}, nworkers + 1)

    exit_criteria = {"gen_max": 501}

    # Perform the run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, libE_specs=libE_specs)

    if is_manager:
        # assert libE_specs['comms'] == 'mpi', 'MPI default comms should be set'
        # Potential to cause a hang
        worker_ids = []
        exp_worker_ids = list(range(1, nworkers + 1))
        mpi_status = MPI.Status()
        for worker in range(1, nworkers + 1):
            worker_rank = world.recv(source=worker, status=mpi_status)
            worker_ids.append(worker_rank)
        assert worker_ids == exp_worker_ids, "MPI World values are not as expected"
        assert len(H) >= 501
        print("\nlibEnsemble with random sampling has generated enough points")
        save_libE_output(H, persis_info, __file__, nworkers)
