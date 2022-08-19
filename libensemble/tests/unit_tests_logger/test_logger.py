#!/usr/bin/env python

"""
Unit test of libensemble log functions.
"""
import os
import logging
from libensemble import logger
from libensemble.comms.logs import LogConfig


def test_set_log_level():
    # Default
    level = logger.get_level()
    assert level == 20, "Log level should be 20. Found: " + str(level)

    logger.set_level("DEBUG")
    level = logger.get_level()
    assert level == 10, "Log level should be 10. Found: " + str(level)

    logger.set_level("WARNING")
    level = logger.get_level()
    assert level == 30, "Log level should be 30. Found: " + str(level)

    logger.set_level("MANAGER_WARNING")
    level = logger.get_level()
    assert level == 35, "Log level should be 35. Found: " + str(level)

    logger.set_level("ERROR")
    level = logger.get_level()
    assert level == 40, "Log level should be 40. Found: " + str(level)

    logger.set_level("INFO")
    level = logger.get_level()
    assert level == 20, "Log level should be 20. Found: " + str(level)


# def test_change_log_level():
#     from libensemble.comms.logs import manager_logging_config
#     manager_logging_config()
#     level_from_config = logger.get_level()
#     assert level_from_config == 20, "Log level from config should be 20. Found: " + str(level)
#     level_from_logger = logging.getLogger('libensemble').getEffectiveLevel()
#     assert level_from_logger == 20, "Log level from logger should be 20. Found: " + str(level)

#     # Now test logger level after change
#     logger.set_level('DEBUG')
#     level_from_logger = logging.getLogger('libensemble').getEffectiveLevel()
#     assert level_from_logger == 10, "Log level from logger should be 10. Found: " + str(level)

#     # Now test inherited logger
#     level_from_child_logger = logging.getLogger('libensemble.libE').getEffectiveLevel()
#     assert level_from_child_logger == 10, "Log level from child logger should be 10. Found: " + str(level)


def test_set_filename():
    from libensemble.comms.logs import manager_logging_config

    alt_name = "alt_name.log"

    logs = LogConfig.config
    assert logs.filename == "ensemble.log", "Log filename expected ensemble.log. Found: " + logs.filename

    logger.set_filename(alt_name)
    assert logs.filename == alt_name, "Log filename expected " + str(alt_name) + ". Found: " + logs.filename

    manager_logging_config()
    logger.set_filename("toolate.log")
    assert logs.filename == alt_name, "Log filename expected " + str(alt_name) + ". Found: " + logs.filename

    assert os.path.isfile(alt_name), "Expected creation of file" + str(alt_name)
    with open(alt_name, "r") as f:
        line = f.readline()
        assert "Cannot set filename after loggers initialized" in line
    try:
        os.remove(alt_name)
    except PermissionError:  # windows only
        pass

    logs = LogConfig.config
    logs.logger_set = True
    logs.set_level("DEBUG")


def test_set_stderr_level():

    stderr_level = logger.get_stderr_level()
    assert stderr_level == 35, "Default stderr copying level is 35, found " + str(stderr_level)

    logger.set_stderr_level("DEBUG")
    stderr_level = logger.get_stderr_level()
    assert stderr_level == 10, "Log level should be 10. Found: " + str(stderr_level)

    logger.set_stderr_level("INFO")
    stderr_level = logger.get_stderr_level()
    assert stderr_level == 20, "Log level should be 20. Found: " + str(stderr_level)

    logger.set_stderr_level("WARNING")
    stderr_level = logger.get_stderr_level()
    assert stderr_level == 30, "Log level should be 30. Found: " + str(stderr_level)

    logger.set_stderr_level("MANAGER_WARNING")
    stderr_level = logger.get_stderr_level()
    assert stderr_level == 35, "Log level should be 35. Found: " + str(stderr_level)

    logger.set_stderr_level("ERROR")
    stderr_level = logger.get_stderr_level()
    assert stderr_level == 40, "Log level should be 40. Found: " + str(stderr_level)

    logger.set_level("ERROR")
    logger_test = logging.getLogger("libensemble")
    logger_test.manager_warning("This test message should not log")


# Need setup/teardown here to kill loggers if running file without pytest
# Issue: cannot destroy loggers and they are set up in other unit tests.
# Partial solution: either rename the file so it is the first unit test, or
#   move this unit test to its own directory.
if __name__ == "__main__":
    test_set_log_level()
    # test_change_log_level()
    test_set_filename()
    test_set_stderr_level()
