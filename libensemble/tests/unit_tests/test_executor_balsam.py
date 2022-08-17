# !/usr/bin/env python
# Integration Test of executor module for libensemble
# Test does not require running full libensemble
import os
import re
import sys
import time
import mock
import pytest
from dataclasses import dataclass

from libensemble.resources.mpi_resources import MPIResourcesException
from libensemble.executors.executor import Executor, ExecutorException, TimeoutExpired
from libensemble.executors.executor import NOT_STARTED_STATES, Application

NCORES = 1
py_startup = "simdir/py_startup.py"

# fake Balsam app
@dataclass
class TestLibeApp:
    site = "libe-unit-test"
    command_template = "python simdir/py_startup.py"

    def sync():
        pass


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
    """Test of registering an App"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    exctr.register_app(TestLibeApp, calc_type="sim", precedent="fake/dir")
    assert isinstance(exctr.apps["python"], Application), \
        "Application object not created based on registered Balsam AppDef"

def test_submit_app_defaults():
    """Test of submitting an App"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job"):
        task = exctr.submit(calc_type="sim")

    assert task in exctr.list_of_tasks, \
        "new task not added to executor's list of tasks"

    assert task == exctr.get_task(task.id), \
        "task retrieved via task ID doesn't match new task"


def test_submit_app_workdir():
    """Test of submitting an App with a workdir"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job"):
        task = exctr.submit(calc_type="sim", workdir="output", machinefile="nope")

    assert task.workdir == os.path.join(exctr.workflow_name, "output"), \
        "workdir not properly defined for new task"

def test_submit_app_dry():
    """Test of dry-run submitting an App"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    exctr = Executor.executor
    task = exctr.submit(calc_type="sim", dry_run=True)

    assert all([task.dry_run, task.done()]), \
        "new task from dry_run wasn't marked as such, or set as done"

def test_submit_app_wait():
    """Test of exctr.submit blocking until app is running"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job") as job:
        job.return_value.state = "RUNNING"
        task = exctr.submit(calc_type="sim", wait_on_start=True)

    assert task.running(), \
        "new task is not marked as running after wait_on_start"

def test_submit_revoke_alloc():
    """Test creating and revoking BatchJob objects through the executor"""
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.BatchJob") as batchjob:
        alloc = exctr.submit_allocation(site_id="libe-unit-test", num_nodes=1, wall_time_min=30)

        assert alloc in exctr.allocations, \
            "batchjob object not appended to executor's list of allocations"

        alloc.scheduler_id = 1
        exctr.revoke_allocation(alloc)

def test_task_wait():
    """Test of killing (cancelling) a balsam app"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job"):
        task = exctr.submit(calc_type="sim")

    task.wait()

def test_task_kill():
    """Test of killing (cancelling) a balsam app"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job"):
        task = exctr.submit(calc_type="sim")

    task.kill()
    assert task.finished and task.state == "USER_KILLED", \
        "task not set as killed after kill method"

if __name__ == "__main__":
    setup_module(__file__)
    test_register_app()
    test_submit_app_defaults()
    test_submit_app_workdir()
    test_submit_app_dry()
    test_submit_app_wait()
    test_submit_revoke_alloc()
    test_task_wait()
    test_task_kill()