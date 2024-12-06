"""
libEnsemble communication interface
====================================================

A comm provides a message-passing abstraction for communication
between a worker user function and the manager.  Basic messages are:

  stop() - manager tells persistent gen/sim to stop
  worker_avail(nworker) - manager tells gen that workers are available
  request(recs) - worker requests simulations
  queued(id) - manager assigns simulation IDs to request
  kill(id) - gen requests manager kill a simulation
  update(id, rec) - manager informs gen of partial sim information
  result(id, rec) - manager informs gen a sim completed
  killed(id) - manager informs gen a sim was killed

To facilitate information sharing, there are also messages for history
access and monitoring (for persistent gens):

  get_history(lo, hi) - gen requests history
  subscribe() - gen subscribes to all history updates

"""

import copy
import queue as thread_queue
from abc import ABC, abstractmethod

# from multiprocessing import Process, Queue, Value, Lock
from multiprocessing import Process, Queue
from threading import Thread
from time import time
from traceback import format_exc


class Timeout(Exception):
    """Communication timeout exception."""


class CommFinishedException(Exception):
    """Read from terminated comm exception."""


class ManagerStop(Exception):
    """Exception raised by default when manager sends a stop message."""


class RemoteException(Exception):
    """Exception raised a remote exception is received."""

    def __init__(self, msg, exc):
        super().__init__(msg)
        self.exc = exc


class CommResult:
    """Container for a result returned on exit."""

    def __init__(self, value):
        self.value = value


class CommResultErr:
    """Container for an exception returned on exit."""

    def __init__(self, msg, exc):
        self.msg = msg
        self.exc = exc


def _timeout_fun(timeout):
    """Return a function that gets timeouts for time remaining."""
    if timeout is None:
        return lambda: None
    else:
        tdeath = time() + timeout
        return lambda: tdeath - time()


class Comm(ABC):
    """Bidirectional communication"""

    @abstractmethod
    def send(self, *args):
        """Send a message."""

    @abstractmethod
    def recv(self, timeout=None):
        """Receive a message or raise TimeoutError."""

    def mail_flag(self):
        """Check whether a message is ready for receipt."""
        return False

    def kill_pending(self):
        """Cancel any pending sends (don't worry about those in the system)."""


class QComm(Comm):
    """Queue-based bidirectional communicator

    A QComm provides a message passing abstraction based on a pair of message
    queues: an inbox for incoming messages and an outbox for outgoing messages.
    These can be used with threads or multiprocessing.
    """

    def __init__(self, inbox, outbox, nworkers=None, copy_msg=False):
        """Set the inbox and outbox queues."""
        self._inbox = inbox
        self._outbox = outbox
        self._copy = copy_msg
        self.recv_buffer = None
        self.nworkers = nworkers

    def get_num_workers(self):
        """Return global _ncomms"""
        return self.nworkers
        # return QComm._ncomms.value

    def send(self, *args):
        """Place a message on the outbox queue."""
        if self._copy:
            args = copy.deepcopy(args)
        self._outbox.put(args)

    def recv(self, timeout=None):
        """Return a message from the inbox queue or raise TimeoutError."""
        pb_result = self.recv_buffer
        self.recv_buffer = None
        if pb_result is not None:
            return pb_result
        try:
            if not self._inbox.empty():
                return self._inbox.get()
            return self._inbox.get(timeout=timeout)
        except thread_queue.Empty:
            raise Timeout()

    # TODO: This should go away once I have internal comms working
    def push_to_buffer(self, *args):
        self.recv_buffer = args

    def mail_flag(self):
        """Check whether a message is ready for receipt."""
        return not self._inbox.empty()


class QCommLocal(Comm):
    def __init__(self, main, *args, **kwargs):
        self._result = None
        self._exception = None
        self._done = False

    def _is_result_msg(self, msg):
        """Return true if message indicates final result (and set result/except)."""
        if len(msg) and isinstance(msg[0], CommResult):
            self._result = msg[0].value
            self._done = True
            return True
        if len(msg) and isinstance(msg[0], CommResultErr):
            self._exception = msg[0]
            self._done = True
            return True
        return False

    def run(self):
        """Start the thread/process."""
        self.handle.start()

    def send(self, *args):
        """Send a message to the thread/process (called from creator)"""
        self.inbox.put(args)

    def recv(self, timeout=None):
        """Return a message from the thread or raise TimeoutError."""
        try:
            if self._done:
                raise CommFinishedException()
            if not self.outbox.empty():
                msg = self.outbox.get()
            else:
                msg = self.outbox.get(timeout=timeout)
            if self._is_result_msg(msg):
                raise CommFinishedException()
            return msg
        except thread_queue.Empty:
            raise Timeout()

    def mail_flag(self):
        """Check whether a message is ready for receipt."""
        return not self.outbox.empty()

    def result(self, timeout=None):
        """Join and return the thread/process main result (or re-raise an exception)."""
        get_timeout = _timeout_fun(timeout)
        while not self._done and (timeout is None or timeout >= 0):
            try:
                msg = self.outbox.get(timeout=timeout)
            except thread_queue.Empty:
                raise Timeout()
            self._is_result_msg(msg)
            timeout = get_timeout()
        if not self._done:
            raise Timeout()
        self.terminate(timeout=timeout)
        if self._exception is not None:
            raise RemoteException(self._exception.msg, self._exception.exc)
        return self._result

    @property
    def running(self):
        """Check if the thread/process is running."""
        return self.handle.is_alive()

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, etype, value, traceback):
        self.handle.join()


def _qcomm_main(comm, main, *args, **kwargs):
    """Main routine -- handles return values and exceptions."""
    try:
        if not kwargs.get("user_function"):
            _result = main(comm, *args, **kwargs)
        else:
            _result = main(*args)
        comm.send(CommResult(_result))
    except Exception as e:
        comm.send(CommResultErr(str(e), format_exc()))
        raise e


class QCommThread(QCommLocal):
    """Launch a user function in a thread with an attached QComm."""

    def __init__(self, main, nworkers, *args, **kwargs):
        self.inbox = thread_queue.Queue()
        self.outbox = thread_queue.Queue()
        super().__init__(self, main, *args, **kwargs)
        comm = QComm(self.inbox, self.outbox, nworkers)
        self.handle = Thread(target=_qcomm_main, args=(comm, main) + args, kwargs=kwargs)

    def terminate(self, timeout=None):
        """Terminate the thread.

        Threads can't easily be killed from the outside. It is possible to achieve such
        functionality, for example, with the `main` function periodically checking some
        variable/output to determine if the function should continue. This is not
        implemented here, so it is currently not possible to terminate the thread when
        calling this method.
        """
        self.handle.join(timeout=timeout)
        if self.running:
            raise Timeout()


class QCommProcess(QCommLocal):
    """Launch a user function in a process with an attached QComm."""

    def __init__(self, main, nworkers, *args, **kwargs):
        self.inbox = Queue()
        self.outbox = Queue()
        super().__init__(self, main, *args, **kwargs)
        self.comm = QComm(self.inbox, self.outbox, nworkers)
        self.handle = Process(target=_qcomm_main, args=(self.comm, main) + args, kwargs=kwargs)

    def terminate(self, timeout=None):
        """Terminate the process."""
        if self.running:
            self.handle.terminate()
        self.handle.join(timeout=timeout)
        if self.running:
            raise Timeout()
