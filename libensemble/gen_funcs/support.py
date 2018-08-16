from mpi4py import MPI

from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, UNSET_TAG, EVAL_GEN_TAG


def sendrecv_mgr_worker_msg(comm, O, status=None):
    """Send message from worker to manager and receive response.
    """
    send_mgr_worker_msg(comm, O)
    return get_mgr_worker_msg(comm, status=status)


def send_mgr_worker_msg(comm, O):
    """Send message from worker to manager.
    """
    D = {'calc_out': O,
         'libE_info': {'persistent': True},
         'calc_status': UNSET_TAG,
         'calc_type': EVAL_GEN_TAG
         }
    comm.send(obj=D, dest=0, tag=EVAL_GEN_TAG)


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
