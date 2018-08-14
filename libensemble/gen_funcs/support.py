from mpi4py import MPI

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP


def get_mgr_worker_msg(comm, status=None):
    """Get message to worker from manager.
    """
    status = status or MPI.Status()
    comm.probe(source=0, tag=MPI.ANY_TAG, status=status)
    tag = status.Get_tag()
    if tag in [STOP_TAG, PERSIS_STOP]:
        return tag, None, None
    Work = comm.recv(buf=None, source=0, tag=MPI.ANY_TAG, status=status)
    calc_in = comm.recv(buf=None, source=0)
    return tag, Work, calc_in
