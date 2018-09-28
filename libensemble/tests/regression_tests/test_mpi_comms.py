import time
from mpi4py import MPI
from libensemble.mpi_comms import MPIComm


def worker_main():
    "Worker main routine"
    comm = MPIComm()
    assert comm.recv() == "Hello"
    assert comm.recv() == "World"
    assert comm.recv() == comm.rank
    comm.send(comm.rank)


def manager_main():
    "Manager main routine"
    worker_comms = [MPIComm(MPI.COMM_WORLD, r)
                    for r in range(1, MPI.COMM_WORLD.Get_size())]
    for comm in worker_comms:
        try:
            okay_flag = True
            comm.recv(0)
            okay_flag = False
        except TimeoutError:
            pass
        assert okay_flag, "Worker comm behaved as expected."
    for comm in worker_comms:
        comm.send("Hello")
        comm.send("World")
        comm.send(comm.remote_rank)
    for comm in worker_comms:
        assert comm.recv(0) == comm.remote_rank


if MPI.COMM_WORLD.Get_rank() == 0:
    manager_main()
else:
    worker_main()
