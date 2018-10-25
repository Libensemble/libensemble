"""
Support for logging over comms
====================================================

The comms mechanism can be used to support consolidated logging at the
manager by an appropriate choice of handlers and filters.  The default
logging behavior is to install a CommLogHandler at each worker, which is
used to pass messages to be handled at the manager, where they are then
selected and emitted.  The WorkerID filter is used to add contextual
information (in the form of a worker field) that identifies the origin of
a given log message (manager or worker ID).
"""

import logging


class CommLogHandler(logging.Handler):
    """Logging handler class that forwards LogRecords to a Comm.
    """

    def __init__(self, comm, pack=None, level=logging.NOTSET):
        "Initialize the handler instance, setting the level and the comm."
        super().__init__(level)
        self.comm = comm
        self.pack = pack

    def emit(self, record):
        "Actually log the record."
        if self.pack is not None:
            self.comm.send(*self.pack(record))
        else:
            self.comm.send(record)


class WorkerIDFilter(logging.Filter):
    """Logging filter to add worker ID to records.
    """

    def __init__(self, worker_id):
        super().__init__()
        self.worker_id = worker_id

    def filter(self, record):
        "Add worker ID to a LogRecord"
        record.worker = getattr(record, 'worker', self.worker_id)
        return True


def worker_logging_config(comm, worker_id=None, logger=None,
                          level=logging.NOTSET):
    """Add a comm handler with worker ID filter to the indicated logger.
    """
    ch = CommLogHandler(comm, pack=lambda rec: (0, rec))
    ch.addFilter(WorkerIDFilter(worker_id or comm.rank))
    logger = logger or logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(ch)


def manager_logging_config(logger=None, fmt=None, filename='ensemble.log',
                           level=logging.DEBUG):
    """Add file-based logging at manager.
    """
    fmt = fmt or '[%(worker)s] %(name)s (%(levelname)s): %(message)s'
    wfilter = WorkerIDFilter(0)
    fh = logging.FileHandler(filename)
    fh.addFilter(wfilter)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root = logger or logging.getLogger()
    root.addHandler(fh)
