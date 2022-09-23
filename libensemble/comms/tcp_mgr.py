"""
TCP-based bidirectional communicator
------------------------------------

"""

from libensemble.comms.comms import QComm
from multiprocessing.managers import BaseManager
from multiprocessing import Queue


class ServerQCommManager:
    """Set up a QComm manager server.

    The QComm manager server provides shared (networked) access to message
    queues for communication between the libensemble manager and workers.
    """

    def __init__(self, port, authkey):
        """Initialize the server on localhost at an indicated TCP port and key."""
        queues = {"shared": Queue()}

        class ServerQueueManager(BaseManager):
            pass

        def get_queue(name):
            if name not in queues:
                queues[name] = Queue()
            return queues[name]

        ServerQueueManager.register("get_queue", callable=get_queue)
        self.manager = ServerQueueManager(address=("", port), authkey=authkey)
        self.manager.start()

    def shutdown(self):
        """Shutdown the manager"""
        self.manager.shutdown()

    @property
    def address(self):
        """Get IP address for socket."""
        return self.manager.address

    def get_queue(self, name):
        """Get a queue from the shared manager"""
        return self.manager.get_queue(name)

    def get_inbox(self, workerID):
        """Get a worker inbox queue."""
        return self.get_queue(f"inbox{workerID}")

    def get_outbox(self, workerID):
        """Get a worker outbox queue."""
        return self.get_queue(f"outbox{workerID}")

    def get_shared(self):
        """Get a shared queue for worker subscription."""
        return self.get_queue("shared")

    def await_workers(self, nworkers):
        """Wait for a pool of workers to join."""
        sharedq = self.get_shared()
        wqueues = []
        for _ in range(nworkers):
            workerID = sharedq.get()
            inbox = self.get_outbox(workerID)
            outbox = self.get_inbox(workerID)
            wqueues.append(QComm(inbox, outbox))
        return wqueues

    def __enter__(self):
        """Context enter."""
        return self

    def __exit__(self, etype, value, traceback):
        """Context exit."""
        self.shutdown()


class ClientQCommManager:
    """Set up a client to the QComm server.

    The client runs at the worker and mediates access to the shared queues
    provided by the server.
    """

    def __init__(self, ip, port, authkey, workerID):
        """Attach by TCP to (ip, port) with a uniquely given workerID"""
        self.workerID = workerID

        class ClientQueueManager(BaseManager):
            pass

        ClientQueueManager.register("get_queue")
        self.manager = ClientQueueManager(address=(ip, port), authkey=authkey)
        self.manager.connect()
        sharedq = self.get_shared()
        sharedq.put(workerID)

    def get_queue(self, name):
        """Get a queue from the server."""
        return self.manager.get_queue(name)

    def get_inbox(self):
        """Get this worker's inbox."""
        return self.get_queue(f"inbox{self.workerID}")

    def get_outbox(self):
        """Get this worker's outbox."""
        return self.get_queue(f"outbox{self.workerID}")

    def get_shared(self):
        """Get the shared queue for worker sign-up."""
        return self.get_queue("shared")

    def __enter__(self):
        """Enter the context."""
        return QComm(self.get_inbox(), self.get_outbox())

    def __exit__(self, etype, value, traceback):
        """Exit the context."""
        pass
