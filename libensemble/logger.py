import logging

from libensemble.comms.logs import LogConfig

# From https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945

MANAGER_WARNING = 35
logging.addLevelName(MANAGER_WARNING, "MANAGER_WARNING")
logging.MANAGER_WARNING = MANAGER_WARNING


def manager_warning(self, message, *args, **kwargs):
    if self.isEnabledFor(MANAGER_WARNING):
        self._log(MANAGER_WARNING, message, args, **kwargs)


logging.Logger.manager_warning = manager_warning
LogConfig(__package__)


def set_level(level):
    """Sets libEnsemble logging level"""
    logs = LogConfig.config
    logs.set_level(level)


def get_level():
    """Returns libEnsemble logging level"""
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
    """Sets logger to mirror certain messages to stderr"""
    logs = LogConfig.config
    logs.set_stderr_level(level)


def get_stderr_level():
    """Returns libEnsemble stderr logging level"""
    logs = LogConfig.config
    return logs.stderr_level
