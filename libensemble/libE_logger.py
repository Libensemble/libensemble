import logging

from libensemble.comms.logs import LogConfig
LogConfig(__package__)


def set_level(level):
    """Set libEnsemble logging level"""
    logs = LogConfig.config
    logs.set_level(level)


def get_level():
    """Return libEnsemble logging level"""
    logs = LogConfig.config
    return logs.log_level


def set_filename(filename):
    """Sets logger filename if loggers not yet created, else None"""
    logs = LogConfig.config
    if logs.logger_set:
        logger = logging.getLogger(logs.name)
        logger.warning("Cannot set filename after loggers initialized")
    else:
        logs.filename = filename


def set_stderr_level(level):
    """ Sets logger to mirror certain messages to stderr"""
    logs = LogConfig.config
    logs.set_stderr_level(level)


def get_stderr_level():
    """ Return libEnsemble stderr logging level """
    logs = LogConfig.config
    return logs.stderr_level
