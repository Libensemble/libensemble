#!/usr/bin/env python

"""
Unit test of libensemble log functions.
"""
import os
from libensemble import libE_logger


def test_set_log_level():
    # Default
    level = libE_logger.get_level()
    assert level == 20, "Log level should be 20. Found: " + str(level)

    libE_logger.set_level('DEBUG')
    level = libE_logger.get_level()
    assert level == 10, "Log level should be 10. Found: " + str(level)

    libE_logger.set_level('WARNING')
    level = libE_logger.get_level()
    assert level == 30, "Log level should be 30. Found: " + str(level)

    libE_logger.set_level('ERROR')
    level = libE_logger.get_level()
    assert level == 40, "Log level should be 40. Found: " + str(level)

    libE_logger.set_level('INFO')
    level = libE_logger.get_level()
    assert level == 20, "Log level should be 20. Found: " + str(level)


# def test_change_log_level():
#     from libensemble.comms.logs import manager_logging_config
#     manager_logging_config()
#     level_from_config = libE_logger.get_level()
#     assert level_from_config == 20, "Log level from config should be 20. Found: " + str(level)
#     level_from_logger = logging.getLogger('libensemble').getEffectiveLevel()
#     assert level_from_logger == 20, "Log level from logger should be 20. Found: " + str(level)

#     # Now test logger level after change
#     libE_logger.set_level('DEBUG')
#     level_from_logger = logging.getLogger('libensemble').getEffectiveLevel()
#     assert level_from_logger == 10, "Log level from logger should be 10. Found: " + str(level)

#     # Now test inherited logger
#     level_from_child_logger = logging.getLogger('libensemble.libE').getEffectiveLevel()
#     assert level_from_child_logger == 10, "Log level from child logger should be 10. Found: " + str(level)


def test_set_filename():
    from libensemble.comms.logs import LogConfig
    from libensemble.comms.logs import manager_logging_config
    alt_name = "alt_name.log"

    # Test
    logs = LogConfig.config
    print('logger set:', logs.logger_set)

    logs = LogConfig.config
    assert logs.filename == "ensemble.log", "Log filename expected ensemble.log. Found: " + logs.filename

    libE_logger.set_filename(alt_name)
    assert logs.filename == alt_name, "Log filename expected " + str(alt_name) + ". Found: " + logs.filename

    manager_logging_config()
    libE_logger.set_filename('toolate.log')
    assert logs.filename == alt_name, "Log filename expected " + str(alt_name) + ". Found: " + logs.filename

    assert os.path.isfile(alt_name), "Expected creation of file" + str(alt_name)
    with open(alt_name, 'r') as f:
        line = f.readline()
        assert "Cannot set filename after loggers initialized" in line
    os.remove(alt_name)

    logs = LogConfig.config
    logs.logger_set = True
    logs.set_level('DEBUG')


# Need setup/teardown here to kill loggers if running file without pytest
# Issue: cannot destroy loggers and they are set up in other unit tests.
# Partial solution: either rename the file so it is the first unit test, or
#   move this unit test to its own directory.
if __name__ == "__main__":
    test_set_log_level()
    # test_change_log_level()
    test_set_filename()
