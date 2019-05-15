# """
# Test of MPI comms.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_mpi_comms.py
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

import sys
from mpi4py import MPI
from libensemble.comms.mpi import MPIComm, Timeout
from libensemble.tests.regression_tests.common import parse_args

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi
# TESTSUITE_NPROCS: 2 4

nworkers, is_master, libE_specs, _ = parse_args()
if libE_specs['comms'] != 'mpi':
    sys.exit("This test can only be run with mpi comms -- aborting...")


def check_recv(comm, expected_msg):
    msg = comm.recv()
    assert msg == expected_msg, "Expected {}, received {}".format(expected_msg, msg)


def worker_main():
    "Worker main routine"
    comm = MPIComm()
    check_recv(comm, "Hello")
    check_recv(comm, "World")
    check_recv(comm, comm.rank)
    comm.send(comm.rank)
    check_recv(comm, "Goodbye")


def manager_main():
    "Manager main routine"
    worker_comms = [
        MPIComm(MPI.COMM_WORLD, r)
        for r in range(1, MPI.COMM_WORLD.Get_size())]
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


if is_master:
    manager_main()
else:
    worker_main()
