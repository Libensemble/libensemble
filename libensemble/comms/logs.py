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
import sys
from libensemble.utils.timer import Timer


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
        self.stat_filename = "libE_stats.txt"
        self.fmt = "[%(worker)s]  %(asctime)s %(name)s (%(levelname)s): %(message)s"
        self.stderr_level = logging.MANAGER_WARNING

    def set_level(self, level):
        """Set logger level either before or after creating loggers"""
        numeric_level = getattr(logging, level.upper(), 10)
        self.log_level = numeric_level
        if self.logger_set:
            logger = logging.getLogger(self.name)
            logger.setLevel(self.log_level)

    def set_stderr_level(self, level):
        """Set logger level for copying messages to stderr"""
        numeric_level = getattr(logging, level.upper(), 30)
        self.stderr_level = numeric_level


class CommLogHandler(logging.Handler):
    """Logging handler class that forwards LogRecords to a Comm."""

    def __init__(self, comm, pack=None, level=logging.NOTSET):
        """Initialize the handler instance, setting the level and the comm."""
        super().__init__(level)
        self.comm = comm
        self.pack = pack

    def emit(self, record):
        """Actually log the record."""
        if self.pack is not None:
            self.comm.send(*self.pack(record))
        else:
            self.comm.send(record)


class WorkerIDFilter(logging.Filter):
    """Logging filter to add worker ID to records."""

    # Give min. width adjustment (uses more space if needs more).
    margin_align = 5

    def __init__(self, worker_id):
        super().__init__()
        self.worker_id = worker_id

        # Prefix used by stats logger
        if worker_id == 0:
            self.prefix = "Manager" + " " * (WorkerIDFilter.margin_align)
        else:
            worker_str = str(self.worker_id).rjust(WorkerIDFilter.margin_align, " ")
            self.prefix = f"Worker {worker_str}"

    def filter(self, record):
        """Add worker ID to a LogRecord"""
        record.worker = getattr(record, "worker", self.worker_id)
        record.prefix = getattr(record, "prefix", self.prefix)
        return True


class ErrorFilter(logging.Filter):
    """Filter to choose messages for stderr of user-defined level"""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        """Confirm messages that exceed specified level"""
        return record.levelno >= self.level


def remove_handlers(logr):
    """Removes all handlers from a logger

    Required, for example, to remove any manager handlers that
    get undesirably given to a worker via 'fork' (from a previous
    libE function call).
    """
    for hdl in logr.handlers[:]:
        logr.removeHandler(hdl)
        hdl.close()


def init_worker_logger(logr, lev):
    """Initialize a worker logger attributes"""
    logr.propagate = False
    logr.setLevel(lev)


def worker_logging_config(comm, worker_id=None):
    """Add a comm handler with worker ID filter to the indicated logger."""
    logconfig = LogConfig.config
    logger = logging.getLogger(logconfig.name)
    slogger = logging.getLogger(logconfig.stats_name)

    ch = CommLogHandler(comm, pack=lambda rec: (0, rec))
    ch.addFilter(WorkerIDFilter(worker_id or comm.rank))

    if logconfig.logger_set:
        remove_handlers(logger)
        remove_handlers(slogger)
    else:
        init_worker_logger(logger, logconfig.log_level)
        init_worker_logger(slogger, logconfig.log_level)
        logconfig.logger_set = True

    logger.addHandler(ch)
    slogger.addHandler(ch)


def manager_logging_config():
    """Add file-based logging at manager."""
    stat_timer = Timer()
    stat_timer.start()

    # Regular logging
    logconfig = LogConfig.config

    if not logconfig.logger_set:
        formatter = logging.Formatter(logconfig.fmt)
        wfilter = WorkerIDFilter(0)
        fh = logging.FileHandler(logconfig.filename, mode="w")
        fh.addFilter(wfilter)
        fh.setFormatter(formatter)
        logger = logging.getLogger(logconfig.name)
        logger.propagate = False
        logger.setLevel(logconfig.log_level)  # Formatter filters on top of this
        logger.addHandler(fh)
        logconfig.logger_set = True

        # Stats logging
        # NB: Could add a specialized handler for immediate flushing
        fhs = logging.FileHandler(logconfig.stat_filename, mode="w")
        fhs.addFilter(wfilter)
        fhs.setFormatter(logging.Formatter("%(prefix)s: %(message)s"))
        stat_logger = logging.getLogger(logconfig.stats_name)
        stat_logger.propagate = False
        stat_logger.setLevel(logging.DEBUG)
        stat_logger.addHandler(fhs)

        # Mirror error-logging to stderr of user-specified level
        fhe = logging.StreamHandler(stream=sys.stderr)
        fhe.addFilter(wfilter)
        efilter = ErrorFilter(logconfig.stderr_level)
        fhe.addFilter(efilter)
        fhe.setFormatter(formatter)
        logger.addHandler(fhe)
    else:
        stat_logger = logging.getLogger(logconfig.stats_name)

    stat_logger.info(f"Starting ensemble at: {stat_timer.date_start}")

    def exit_logger():
        stat_timer.stop()
        stat_logger.info(f"Exiting ensemble at: {stat_timer.date_end} Time Taken: {stat_timer.elapsed}")

        # If closing logs - each libE() call will log to a new file.
        # fh.close()
        # fhs.close()

    return exit_logger
