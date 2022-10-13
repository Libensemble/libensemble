"""
Test of MPI comms.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python test_mpi_comms.py

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

from mpi4py import MPI
from libensemble.comms.mpi import MPIComm, Timeout
from libensemble.tools import parse_args

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4

# Main block is necessary only when using local comms with spawn start method (default on macOS and Windows).
if __name__ == "__main__":

    nworkers, is_manager, libE_specs, _ = parse_args()

    assert libE_specs["comms"] == "mpi", "This test can only be run with mpi comms -- aborting..."

    def check_recv(comm, expected_msg):
        msg = comm.recv()
        assert msg == expected_msg, f"Expected {expected_msg}, received {msg}"

    def worker_main(mpi_comm):
        "Worker main routine"
        comm = MPIComm(mpi_comm)
        check_recv(comm, "Hello")
        check_recv(comm, "World")
        check_recv(comm, comm.rank)
        comm.send(comm.rank)
        check_recv(comm, "Goodbye")

    def manager_main(mpi_comm):
        "Manager main routine"
        worker_comms = [MPIComm(mpi_comm, r) for r in range(1, mpi_comm.Get_size())]
        for comm in worker_comms:
            try:
                okay_flag = True
                comm.recv(0)
                okay_flag = False
            except Timeout:
                pass
            assert okay_flag, "Worker comm behaved as expected."
        for comm in worker_comms:
            comm.send("Hello")
            comm.send("World")
            comm.send(comm.remote_rank)
        for comm in worker_comms:
            check_recv(comm, comm.remote_rank)
            comm.send("Goodbye")

    def mpi_comm_excl(exc=[0]):
        world_group = MPI.COMM_WORLD.Get_group()
        new_group = world_group.Excl(exc)
        mpi_comm = MPI.COMM_WORLD.Create(new_group)
        return mpi_comm

    def check_ranks(mpi_comm, test_exp, test_num):
        try:
            rank = mpi_comm.Get_rank()
        except Exception:
            rank = -1
        comm_ranks_in_world = MPI.COMM_WORLD.allgather(rank)
        print(f"got {comm_ranks_in_world},  exp {test_exp[test_num]} ", flush=True)
        # This is really testing the test is testing what is it supposed to test
        assert comm_ranks_in_world == test_exp[test_num], (
            "comm_ranks_in_world are: " + str(comm_ranks_in_world) + " Expected: " + str(test_exp[test_num])
        )
        if rank == -1:
            return False
        return True

    # Run Tests
    all_ranks = list(range(MPI.COMM_WORLD.Get_size()))

    tests = {
        1: MPI.COMM_WORLD.Dup,
        2: mpi_comm_excl,
    }

    test_exp = {
        1: all_ranks,
        2: [-1] + all_ranks[:-1],
    }

    for test_num in range(1, len(tests) + 1):
        mpi_comm = tests[test_num]()
        if check_ranks(mpi_comm, test_exp, test_num):
            is_manager = mpi_comm.Get_rank() == 0
            if is_manager:
                manager_main(mpi_comm)
            else:
                worker_main(mpi_comm)
