# !/usr/bin/env python
# Integration Test of executor module for libensemble
# Test does not require running full libensemble
import os
import re
import sys
import time
import mock
import pytest
import socket

from balsam.api import ApplicationDefinition

from libensemble.resources.mpi_resources import MPIResourcesException
from libensemble.executors.executor import Executor, ExecutorException, TimeoutExpired
from libensemble.executors.executor import NOT_STARTED_STATES, Application

NCORES = 1
py_startup = "simdir/py_startup.py"

class TestLibeApp(ApplicationDefinition):
    site = "libe-unit-test"
    command_template = "python simdir/py_startup.py"


def setup_module(module):
    try:
        print("setup_module module:%s" % module.__name__)
    except AttributeError:
        print("setup_module (direct run) module:%s" % module)
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def teardown_module(module):
    try:
        print("teardown_module module:%s" % module.__name__)
    except AttributeError:
        print("teardown_module (direct run) module:%s" % module)
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


# This would typically be in the user calling script
def setup_executor():
    """Set up a Balsam Executor with sim app"""
    from libensemble.executors.balsam_executors import BalsamExecutor
    exctr = BalsamExecutor()



# Tests ========================================================================================

def test_register_app():
    """Test of registering an ApplicationDefinition"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    exctr.register_app(TestLibeApp, calc_type="sim", precedent="fake/dir")
    assert isinstance(exctr.apps["python"], Application), \
        "Application object not created based on registered Balsam AppDef"

def test_submit_app_defaults():
    """Test of submitting an ApplicationDefinition"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    exctr = Executor.executor


def test_submit_app_workdir():
    pass

def test_submit_app_dry():
    pass

def test_submit_app_wait():
    pass

def test_submit_alloc():
    pass

def test_revoke_alloc():
    pass

def test_task_timing():
    pass

def test_task_poll():
    pass

def test_task_wait():
    pass

def test_task_kill():
    pass

if __name__ == "__main__":
    setup_module(__file__)
    test_register_app()
    test_submit_app_defaults()
    test_submit_app_workdir()
    test_submit_app_dry()
    test_submit_app_wait()
    test_submit_alloc()
    test_revoke_alloc()
    test_task_timing()
    test_task_poll()
    test_task_wait()
    test_task_kill()