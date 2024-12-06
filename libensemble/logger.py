import logging

from libensemble.comms.logs import LogConfig

# From https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945

MANAGER_WARNING = 35
logging.addLevelName(MANAGER_WARNING, "MANAGER_WARNING")
logging.MANAGER_WARNING = MANAGER_WARNING


def manager_warning(self, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(MANAGER_WARNING):
        self._log(MANAGER_WARNING, message, args, **kwargs)


logging.Logger.manager_warning = manager_warning

VDEBUG = 5
logging.addLevelName(VDEBUG, "VDEBUG")
logging.VDEBUG = VDEBUG


def vdebug(self, message, *args, **kwargs):
    if self.isEnabledFor(VDEBUG):
        self._log(VDEBUG, message, args, **kwargs)


logging.Logger.vdebug = vdebug

LogConfig(__package__)


def set_level(level: int) -> None:
    """Sets libEnsemble logging level"""
    logs = LogConfig.config
    logs.set_level(level)


def get_level() -> int:
    """Returns libEnsemble logging level"""
    logs = LogConfig.config
    return logs.log_level


def set_filename(filename: str) -> None:
    """Sets logger filename if loggers not yet created, else None"""
    logs = LogConfig.config
    if logs.logger_set:
        logger = logging.getLogger(logs.name)
        logger.warning("Cannot set filename after loggers initialized")
    else:
        logs.filename = filename


def set_directory(dirname: str) -> None:
    """Sets target directory to contain logfiles if loggers not yet created"""
    logs = LogConfig.config
    logs.set_directory(dirname)


def set_stderr_level(level: int) -> None:
    """Sets logger to mirror certain messages to stderr"""
    logs = LogConfig.config
    logs.set_stderr_level(level)


def get_stderr_level() -> int:
    """Returns libEnsemble stderr logging level"""
    logs = LogConfig.config
    return logs.stderr_level
