#!/usr/bin/env python

"""
Unit test of libensemble log functions.
"""

import logging
from libensemble import libE_logger

def test_setlogger():
    
    libE_logger.set_level('INFO')
    level = logging.getLogger('libensemble').getEffectiveLevel()
    assert level==20, "Log level should be 20. Found: " + str(level)

    libE_logger.set_level('WARNING')
    level = logging.getLogger('libensemble').getEffectiveLevel()
    assert level==30, "Log level should be 30. Found: " + str(level)

    libE_logger.set_level('ERROR')
    level = logging.getLogger('libensemble').getEffectiveLevel()
    assert level==40, "Log level should be 40. Found: " + str(level)

    libE_logger.set_level('DEBUG')
    level = logging.getLogger('libensemble').getEffectiveLevel()
    assert level==10, "Log level should be 10. Found: " + str(level)

if __name__ == "__main__":
    test_setlogger()
