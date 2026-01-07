#!/usr/bin/env python

"""
Unit test of libensemble log functions.
"""
import logging
import os

import pytest

from libensemble import logger
from libensemble.comms.logs import LogConfig


def test_set_log_level():
    # Default
    level = logger.get_level()
    assert level == 20, "Log level should be 20. Found: " + str(level)

    logger.set_level("VDEBUG")
    level = logger.get_level()
    assert level == 5, "Log level should be 5. Found: " + str(level)

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
        file_content = f.read()
        assert "Cannot set filename after loggers initialized" in file_content
    try:
        os.remove(alt_name)
    except PermissionError:  # windows only
        pass

    logs = LogConfig.config
    logs.logger_set = True
    logs.set_level("DEBUG")


def test_set_directory(tmp_path):
    logs = LogConfig.config
    logs.logger_set = False
    logs.filename = "ensemble.log"
    logs.stat_filename = "libE_stats.txt"

    logger.set_directory(tmp_path)
    assert logs.filename == os.path.join(tmp_path, "ensemble.log")
    assert logs.stat_filename == os.path.join(tmp_path, "libE_stats.txt")


def test_set_stderr_level():
    stderr_level = logger.get_stderr_level()
    assert stderr_level == 35, "Default stderr copying level is 35, found " + str(stderr_level)

    logger.set_stderr_level("VDEBUG")
    stderr_level = logger.get_stderr_level()
    assert stderr_level == 5, "Log level should be 5. Found: " + str(stderr_level)

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


def test_custom_log_levels():
    from libensemble.comms.logs import manager_logging_config

    manager_logging_config()
    logger_test = logging.getLogger("libensemble")
    logger.set_level("ERROR")
    logger_test.manager_warning("This test message should not log")
    logger.set_level("DEBUG")
    logger_test.vdebug("This test message should not log")
    logger.set_level("VDEBUG")
    logger_test.manager_warning("This manager_warning message should log")
    logger_test.vdebug("This vdebug message should log")

    with open(LogConfig.config.filename, "r") as f:
        file_content = f.read()
        assert "This manager_warning message should log" in file_content
        assert "This vdebug message should log" in file_content
        assert "This test message should not log" not in file_content


# Need setup/teardown here to kill loggers if running file without pytest
# Issue: cannot destroy loggers and they are set up in other unit tests.
# Partial solution: either rename the file so it is the first unit test, or
#   move this unit test to its own directory.
if __name__ == "__main__":
    pytest.main([__file__])
