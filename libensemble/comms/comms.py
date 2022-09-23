"""
libEnsemble communication interface
====================================================

A comm provides a message passing abstraction for communication
between a worker user function and the manager.  Basic messages are:

  stop() - manager tells persistent gen/sim to stop
  worker_avail(nworker) - manager tells gen that workers are available
  request(recs) - worker requests simulations
  queued(id) - manager assigns simulation IDs to request
  kill(id) - gen requests manager kill a simulation
  update(id, rec) - manager informs gen of partial sim information
  result(id, rec) - manager informs gen a sim completed
  killed(id) - manager informs gen a sim was killed

To facilitate information sharing, we also have messages for history
access and monitoring (for persistent gens):

  get_history(lo, hi) - gen requests history
  subscribe() - gen subscribes to all history updates

"""

from abc import ABC, abstractmethod
from time import time
from threading import Thread

# from multiprocessing import Process, Queue, Value, Lock
from multiprocessing import Process, Queue
from traceback import format_exc
import queue
import copy

import numpy as np


class Timeout(Exception):
    """Communication timeout exception."""


class CommFinishedException(Exception):
    """Read from terminated comm exception."""


class ManagerStop(Exception):
    """Exception raised by default when manager sends a stop message."""


class RemoteException(Exception):
    """Exception raised when we received a remote exception."""

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
        """Check whether we know a message is ready for receipt."""
        return False

    def kill_pending(self):
        """Cancel any pending sends (don't worry about those in the system)."""


class QComm(Comm):
    """Queue-based bidirectional communicator

    A QComm provides a message passing abstraction based on a pair of message
    queues: an inbox for incoming messages and an outbox for outgoing messages.
    These can be used with threads or multiprocessing.
    """

    # Integer count  - shared amongst processes
    # Supports adding/removing workers - only works with 'fork'
    # lock = Lock()
    # _ncomms = Value("i", 0)

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
        except queue.Empty:
            raise Timeout()

    # TODO: This should go away once I have internal comms working
    def push_to_buffer(self, *args):
        self.recv_buffer = args

    def mail_flag(self):
        """Check whether we know a message is ready for receipt."""
        return not self._inbox.empty()


class QCommThread(Comm):
    """Launch a user function in a thread with an attached QComm."""

    def __init__(self, main, nworkers, *args, **kwargs):
        self.inbox = queue.Queue()
        self.outbox = queue.Queue()
        self.main = main
        self._result = None
        self._exception = None
        kwargs["comm"] = QComm(self.inbox, self.outbox, nworkers, True)
        self.thread = Thread(target=self._qcomm_main, args=args, kwargs=kwargs)

    def send(self, *args):
        """Send a message to the thread (called from creator)"""
        self.inbox.put(copy.deepcopy(args))

    def recv(self, timeout=None):
        """Return a message from the thread or raise TimeoutError."""
        try:
            if not self.outbox.empty():
                return self.outbox.get()
            return self.outbox.get(timeout=timeout)
        except queue.Empty:
            raise Timeout()

    def mail_flag(self):
        """Check whether we know a message is ready for receipt."""
        return not self.outbox.empty()

    def run(self):
        """Start the thread."""
        self.thread.start()

    def result(self):
        """Join and return the thread main result (or re-raise an exception)."""
        self.thread.join()
        if isinstance(self._exception, Exception):
            raise self._exception
        return self._result

    @property
    def running(self):
        """Check if the thread is running."""
        return self.thread.is_alive()

    def _qcomm_main(self, *args, **kwargs):
        """Main routine -- handles return values and exceptions."""
        try:
            self._result = self.main(*args, **kwargs)
        except Exception as e:
            self._exception = e

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, etype, value, traceback):
        self.thread.join()


class QCommProcess(Comm):
    """Launch a user function in a process with an attached QComm."""

    def __init__(self, main, nworkers, *args, **kwargs):
        self.inbox = Queue()
        self.outbox = Queue()
        self._result = None
        self._exception = None
        self._done = False
        comm = QComm(self.inbox, self.outbox, nworkers)

        # with QComm.lock:
        #     QComm._ncomms.value += 1

        self.process = Process(target=QCommProcess._qcomm_main, args=(comm, main) + args, kwargs=kwargs)

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

    def send(self, *args):
        """Send a message to the thread (called from creator)"""
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
        except queue.Empty:
            raise Timeout()

    def mail_flag(self):
        """Check whether we know a message is ready for receipt."""
        return not self.outbox.empty()

    def run(self):
        """Start the process."""
        self.process.start()

    def result(self, timeout=None):
        """Join and return the thread main result (or re-raise an exception)."""
        get_timeout = _timeout_fun(timeout)
        while not self._done and (timeout is None or timeout >= 0):
            try:
                msg = self.outbox.get(timeout=timeout)
            except queue.Empty:
                raise Timeout()
            self._is_result_msg(msg)
            timeout = get_timeout()
        if not self._done:
            raise Timeout()
        self.process.join(timeout=timeout)
        if self.running:
            raise Timeout()
        if self._exception is not None:
            raise RemoteException(self._exception.msg, self._exception.exc)
        # with QComm.lock:
        #     QComm._ncomms.value -= 1
        return self._result

    def terminate(self, timeout=None):
        """Terminate the process."""
        if self.running:
            self.process.terminate()
        self.process.join(timeout=timeout)
        if self.running:
            raise Timeout()
        # with QComm.lock:
        #     QComm._ncomms.value -= 1

    @property
    def running(self):
        """Return true if process is running"""
        return self.process.is_alive()

    @staticmethod
    def _qcomm_main(comm, main, *args, **kwargs):
        """Main routine -- handles return values and exceptions."""
        try:
            _result = main(comm, *args, **kwargs)
            comm.send(CommResult(_result))
        except Exception as e:
            comm.send(CommResultErr(str(e), format_exc()))
            raise e

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, etype, value, traceback):
        self.process.join()


class CommHandler(ABC):
    """Comm wrapper with message handler dispatching.

    The comm wrapper defines a message processor that dispatches to
    different handler methods based on message types.  An incoming message
    with the tag 'foo' gets dispatched to a handler 'on_foo'; if 'on_foo'
    is not defined, we pass to the 'on_unhandled_message' routine.
    """

    def __init__(self, comm):
        """Set the comm to be wrapped."""
        self.comm = comm

    def send(self, *args):
        """Send via the comm."""
        self.comm.send(*args)

    def process_message(self, timeout=None):
        """Receive and process a message via the comm."""
        msg = self.comm.recv(timeout)
        msg_type = msg[0]
        args = msg[1:]
        try:
            method = f"on_{msg_type}"
            handler = getattr(self, method)
        except AttributeError:
            return self.on_unhandled_message(msg)
        return handler(*args)

    def on_unhandled_message(self, msg):
        """Handle any messages for which there are no named handlers."""
        raise ValueError(f"No handler available for message {msg[0]}{msg[1:]}")


class GenCommHandler(CommHandler):
    """Wrapper for handling messages at a persistent gen."""

    def send_request(self, recs):
        """Request new evaluations."""
        self.send("request", recs)

    def send_kill(self, sim_id):
        """Kill an evaluation."""
        self.send("kill", sim_id)

    def send_get_history(self, lo, hi):
        """Request history from manager."""
        self.send("get_history", lo, hi)

    def send_subscribe(self):
        """Request subscription to updates on sims not launched by this gen."""
        self.send("subscribe")

    def on_stop(self):
        """Handle stop message."""
        raise ManagerStop()

    @abstractmethod
    def on_worker_avail(self, nworker):
        """Handle updated number of workers available to perform sims."""

    @abstractmethod
    def on_queued(self, sim_id):
        """Handle sim_id assignment in response to a request"""

    @abstractmethod
    def on_result(self, sim_id, recs):
        """Handle simulation results"""

    @abstractmethod
    def on_update(self, sim_id, recs):
        """Handle simulation updates"""

    @abstractmethod
    def on_killed(self, sim_id):
        """Handle a simulation kill"""


class SimCommHandler(CommHandler):
    """Wrapper for handling messages at sim."""

    def send_result(self, sim_id, recs):
        """Send a simulation result"""
        self.send("result", sim_id, recs)

    def send_update(self, sim_id, recs):
        """Send a simulation update"""
        self.send("update", sim_id, recs)

    def send_killed(self, sim_id):
        """Send notification that a simulation was killed"""
        self.send("killed", sim_id)

    def on_stop(self):
        """Handle stop message."""
        raise ManagerStop()

    @abstractmethod
    def on_request(self, sim_id, recs):
        """Handle a request for a simulation"""

    @abstractmethod
    def on_kill(self, sim_id):
        """Handle a request to kill a simulation"""


class CommEval(GenCommHandler):
    """Future-based interface for generator comms"""

    def __init__(self, comm, workers=0, gen_specs=None):
        super().__init__(comm)
        self.sim_started = 0
        self.sim_pending = 0
        self.workers = workers
        self.gen_specs = gen_specs
        self.promises = {}
        self.returning_promises = None
        self.waiting_for_queued = 0

    def request(self, recs):
        """Request simulations, return promises"""
        self.sim_started += len(recs)
        self.sim_pending += len(recs)
        self.send_request(recs)
        self.waiting_for_queued = len(recs)
        while self.waiting_for_queued > 0:
            self.process_message()
        returning_promises = self.returning_promises
        self.returning_promises = None
        return returning_promises

    def __call__(self, *args, **kwargs):
        """Request a simulation and return a promise"""
        assert not (args and kwargs), "Must specify simulation args by position or keyword, but not both"
        assert args or kwargs, "Must specify simulation arguments."
        rec = np.zeros(1, dtype=self.gen_specs["out"])
        if args:
            assert len(args) == len(self.gen_specs["out"]), "Wrong number of positional arguments in sim call."
            for k, spec in enumerate(self.gen_specs["out"]):
                name = spec[0]
                rec[name] = args[k]
        else:
            for name, value in kwargs.items():
                rec[name] = value
        return self.request(rec)[0]

    def wait_any(self):
        """Wait for any pending simulation to be done"""
        sim_pending = self.sim_pending
        while sim_pending == self.sim_pending:
            self.process_message()

    def wait_all(self):
        """Wait for all pending simulations to be done"""
        while self.sim_pending > 0:
            self.process_message()

    # --- Message handlers

    def on_worker_avail(self, nworker):
        """Update worker count"""
        self.workers = nworker
        return -1

    def on_queued(self, sim_id):
        """Set up futures with indicated simulation IDs"""
        lo = sim_id
        hi = sim_id + self.waiting_for_queued
        self.waiting_for_queued = 0
        self.returning_promises = []
        for s in range(lo, hi):
            promise = Future(self, s)
            self.promises[s] = promise
            self.returning_promises.append(promise)
        return -1

    def on_result(self, sim_id, recs):
        """Handle completed simulation"""
        for k, rec in enumerate(recs):
            self.sim_pending -= 1
            self.promises[sim_id + k].on_result(rec)
        return sim_id

    def on_update(self, sim_id, recs):
        """Handle updated simulation"""
        for k, rec in enumerate(recs):
            self.promises[sim_id + k].on_update(rec)
        return sim_id

    def on_killed(self, sim_id):
        """Handle killed simulation"""
        self.sim_pending -= 1
        self.promises[sim_id].on_killed()
        return sim_id


class Future:
    """Future objects for monitoring asynchronous simulation calls.

    The Future objects are not meant to be instantiated on their own;
    they are only produced by a call on a CommEval object.
    """

    def __init__(self, ceval, sim_id):
        self._ceval = ceval
        self._id = sim_id
        self._comm = ceval.comm
        self._result = None
        self._killed = False
        self._success = False

    @property
    def current_result(self):
        """Return the current (possibly incomplete) result immediately."""
        return self._result

    def cancelled(self):
        """Return True if the simulation was killed."""
        return self._killed

    def done(self):
        """Return True if the simulation completed successfully or was killed."""
        return self._success or self._killed

    def cancel(self):
        """Cancel the simulation."""
        self._ceval.send_kill(self._id)

    def result(self, timeout=None):
        """Get the result of the simulation or throw a timeout."""
        while not self.done():
            if timeout is not None and timeout < 0:
                raise Timeout()
            tstart = time()
            try:
                self._ceval.process_message(timeout)
            except Timeout:
                pass
            if timeout is not None:
                timeout -= time() - tstart
        return self._result

    # --- Message handlers

    def on_result(self, result):
        """Handle an incoming result."""
        self._result = result
        self._success = True

    def on_update(self, result):
        """Handle an incoming update."""
        self._result = result

    def on_killed(self):
        """Handle a kill notification."""
        self._killed = True
