"""
MPI-based bidirectional communicator
------------------------------------

"""

import time
from mpi4py import MPI
from libensemble.comms.comms import Comm, Timeout


class MPIComm(Comm):
    """MPI-based bidirectional communicator.

    The MPIComm object provides the same message queue pair abstraction as the
    other comm objects.  In order to provide nonblocking queue semantics
    (i.e. send returns immediately after putting the message in an internal
    queue rather than after it is 'in the system'), we use the MPI isend.
    Because of how mpi4py is implemented, the isend call immediately creates
    a pickle in a separate buffer, which is associated with the request
    object, and so will not be garbage-collected until the request object
    goes away.  We keep an internal _outbox list to track all pending
    request objects; if the MPIComm is ever garbage-collected while requests
    are still pending, we cancel those requests.
    """

    def __init__(self, mpi_comm=MPI.COMM_WORLD, remote_rank=0):
        "Initialize with a given MPI communicator and rank for the other end"
        self.mpi_comm = mpi_comm
        self.remote_rank = remote_rank
        self.status = MPI.Status()
        self._outbox = []
        self._pushed = None

    def __del__(self):
        "Wait on anything pending if comm is killed."
        for req in self._outbox:
            req.Wait()

    def mail_flag(self):
        return (self._pushed is not None
                or self.mpi_comm.Iprobe(source=self.remote_rank))

    def kill_pending(self):
        "Make sure pending requests are cancelled if the comm is killed."
        for req in self._outbox:
            if not req.Test():
                req.Cancel()
        self._outbox = []

    @property
    def rank(self):
        return self.mpi_comm.Get_rank()

    def clean_outbox(self):
        "Discard the request objects for any completed isends"
        self._outbox = [req for req in self._outbox if not req.Test()]

    def send(self, *args):
        "Send the requested message (as a pickle) via an MPI isend"
        self.clean_outbox()
        msg, tag = self.process_outgoing(args)
        req = self.mpi_comm.isend(msg, dest=self.remote_rank, tag=tag)
        self._outbox.append(req)

    def recv(self, timeout=None):
        "Receive a message or raise TimeoutError."
        if self._pushed is not None:
            result = self._pushed
            self._pushed = None
            return result
        if timeout is not None:
            tfinal = time.time() + timeout
            while not self.mail_flag():
                if time.time() > tfinal:
                    raise Timeout()
        result = self.mpi_comm.recv(source=self.remote_rank, status=self.status)
        return self.process_incoming(result, self.status)

    def process_outgoing(self, msg):
        "Convert a communicator-format message to an MPI message and tag."
        return msg, 0

    def process_incoming(self, msg, status):
        "Convert an MPI message and tag to a local communicator format message."
        return msg[0]

    def push_back(self, *args):
        assert self._pushed is None, "Cannot push back multiple messages"
        self._pushed = args

    def get_num_workers(self):
        return self.mpi_comm.Get_size() - 1


class MainMPIComm(MPIComm):
    """MPI communicator used by the workers and managers for the moment."""

    def process_incoming(self, msg, status):
        return status.Get_tag(), msg

    def process_outgoing(self, msg):
        return msg[1], msg[0]
