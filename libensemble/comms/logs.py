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


class LogConfig:
    """Class for storing logging configuration info"""
    config = None

    def __init__(self, name):
        """Instantiate a new LogConfig instance."""
        LogConfig.config = self
        self.logger_set = False
        self.log_level = logging.INFO
        self.name = name
        self.stats_name = name + ".calc stats"
        self.filename = "ensemble.log"
        self.stat_filename = 'libE_stats.txt'
        self.fmt = '[%(worker)s] %(name)s (%(levelname)s): %(message)s'

    def set_level(self, level):
        """Set logger level either before or after creating loggers"""
        numeric_level = getattr(logging, level.upper(), 10)
        self.log_level = numeric_level
        if self.logger_set:
            logger = logging.getLogger(self.name)
            logger.setLevel(self.log_level)


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


def worker_logging_config(comm, worker_id=None):
    """Add a comm handler with worker ID filter to the indicated logger.
    """
    logconfig = LogConfig.config
    if not logconfig.logger_set:
        ch = CommLogHandler(comm, pack=lambda rec: (0, rec))
        ch.addFilter(WorkerIDFilter(worker_id or comm.rank))
        logger = logging.getLogger(logconfig.name)
        logger.propagate = False
        logger.setLevel(logconfig.log_level)
        logger.addHandler(ch)
        logconfig.logger_set = True


def manager_logging_config():
    """Add file-based logging at manager.
    """

    # Regular logging
    logconfig = LogConfig.config
    if not logconfig.logger_set:
        formatter = logging.Formatter(logconfig.fmt)
        wfilter = WorkerIDFilter(0)
        fh = logging.FileHandler(logconfig.filename)
        fh.addFilter(wfilter)
        fh.setFormatter(formatter)
        logger = logging.getLogger(logconfig.name)
        logger.propagate = False
        logger.setLevel(logconfig.log_level)  # Formatter filters on top of this
        logger.addHandler(fh)
        logconfig.logger_set = True

        # Stats logging
        # NB: Could add a specialized handler for immediate flushing
        fh = logging.FileHandler(logconfig.stat_filename, mode='w')
        fh.addFilter(wfilter)
        fh.setFormatter(logging.Formatter('Worker %(worker)5d: %(message)s'))
        stat_logger = logging.getLogger(logconfig.stats_name)
        stat_logger.propagate = False
        stat_logger.setLevel(logging.DEBUG)
        stat_logger.addHandler(fh)
