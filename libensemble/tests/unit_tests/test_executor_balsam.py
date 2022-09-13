# !/usr/bin/env python
# Integration Test of executor module for libensemble
# Test does not require running full libensemble
import os
import sys
import mock
import pytest
import datetime
from dataclasses import dataclass

from libensemble.executors.executor import (
    Executor,
    Application,
    ExecutorException,
    TimeoutExpired,
)


# fake Balsam app
class TestLibeApp:
    site = "libe-unit-test"
    command_template = "python simdir/py_startup.py"

    def sync():
        pass


# fake EventLog object
@dataclass
class LogEventTest:
    timestamp: datetime.datetime = None


def setup_module(module):
    try:
        print(f"setup_module module:{module.__name__}")
    except AttributeError:
        print(f"setup_module (direct run) module:{module}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def teardown_module(module):
    try:
        print(f"teardown_module module:{module.__name__}")
    except AttributeError:
        print(f"teardown_module (direct run) module:{module}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


# This would typically be in the user calling script
def setup_executor():
    """Set up a Balsam Executor with sim app"""
    from libensemble.executors.balsam_executors import BalsamExecutor

    exctr = BalsamExecutor()  # noqa F841


# Tests ========================================================================================


@pytest.mark.extra
def test_register_app():
    """Test of registering an App"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor

    exctr.serial_setup()  # does nothing, compatibility with legacy-balsam-exctr
    exctr.add_app("hello", "world")  # does nothing, compatibility with legacy-balsam-exctr
    exctr.set_resources("hello")  # does nothing, compatibility with other executors

    exctr.register_app(TestLibeApp, calc_type="sim", precedent="fake/dir")
    assert isinstance(
        exctr.apps["python"], Application
    ), "Application object not created based on registered Balsam AppDef"

    exctr.register_app(TestLibeApp, app_name="test")
    assert isinstance(
        exctr.apps["test"], Application
    ), "Application object not created based on registered Balsam AppDef"


@pytest.mark.extra
def test_submit_app_defaults():
    """Test of submitting an App"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job"):
        task = exctr.submit(calc_type="sim")
        task = exctr.submit(app_name="test")

    assert task in exctr.list_of_tasks, "new task not added to executor's list of tasks"

    assert task == exctr.get_task(task.id), "task retrieved via task ID doesn't match new task"

    with pytest.raises(ExecutorException):
        task = exctr.submit()
        pytest.fail("Expected exception")


@pytest.mark.extra
def test_submit_app_workdir():
    """Test of submitting an App with a workdir"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job"):
        task = exctr.submit(calc_type="sim", workdir="output", machinefile="nope")

    assert task.workdir == os.path.join(exctr.workflow_name, "output"), "workdir not properly defined for new task"


@pytest.mark.extra
def test_submit_app_dry():
    """Test of dry-run submitting an App"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    task = exctr.submit(calc_type="sim", dry_run=True)
    task.poll()

    assert all([task.dry_run, task.done()]), "new task from dry_run wasn't marked as such, or set as done"


@pytest.mark.extra
def test_submit_app_wait():
    """Test of exctr.submit blocking until app is running"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job") as job:
        with mock.patch("libensemble.executors.balsam_executors.balsam_executor.EventLog") as log:
            job.return_value.state = "RUNNING"
            log.objects.filter.return_value = [
                LogEventTest(timestamp=datetime.datetime(2022, 4, 21, 20, 29, 33, 455144))
            ]
            task = exctr.submit(calc_type="sim", wait_on_start=True)
            assert task.running(), "new task is not marked as running after wait_on_start"

            log.objects.filter.return_value = [LogEventTest(timestamp=None)]
            task = exctr.submit(calc_type="sim", wait_on_start=True)
            assert task.runtime == 0, "runtime should be 0 without Balsam timestamp evaluated"


@pytest.mark.extra
def test_submit_revoke_alloc():
    """Test creating and revoking BatchJob objects through the executor"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.BatchJob"):
        alloc = exctr.submit_allocation(site_id="libe-unit-test", num_nodes=1, wall_time_min=30)

        assert alloc in exctr.allocations, "batchjob object not appended to executor's list of allocations"

        alloc.scheduler_id = None
        assert not exctr.revoke_allocation(
            alloc, timeout=3
        ), "unable to revoke allocation if Balsam never returns scheduler ID"

        alloc.scheduler_id = 1
        assert exctr.revoke_allocation(
            alloc, timeout=3
        ), "should've been able to revoke allocation if scheduler ID available"


@pytest.mark.extra
def test_task_poll():
    """Test of killing (cancelling) a balsam app"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job") as job:
        with mock.patch("libensemble.executors.balsam_executors.balsam_executor.EventLog"):
            task = exctr.submit(calc_type="sim")

            job.return_value.state = "PREPROCESSED"
            task.poll()
            assert task.state == "WAITING", "task should've been considered waiting based on balsam state"

            job.return_value.state = "FAILED"
            task.poll()
            assert task.state == "FAILED", "task should've been considered failed based on balsam state"

            task = exctr.submit(calc_type="sim")

            job.return_value.state = "JOB_FINISHED"
            task.poll()
            assert task.state == "FINISHED", "task was not finished after wait method"

    assert not task.running(), "task shouldn't be running after wait method returns"

    assert task.done(), "task should be 'done' after wait method"


@pytest.mark.extra
def test_task_wait():
    """Test of killing (cancelling) a balsam app"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job") as job:
        with mock.patch(
            "libensemble.executors.balsam_executors.balsam_executor.EventLog"
        ):  # need to patch since wait polls
            task = exctr.submit(calc_type="sim")

            job.return_value.state = "RUNNING"
            with pytest.raises(TimeoutExpired):
                task.wait(timeout=3)
                pytest.fail("Expected exception")

            job.return_value.state = "JOB_FINISHED"
            task.wait(timeout=3)
            task.wait(timeout=3)  # should return immediately since self._check_poll() should return False
            assert task.state == "FINISHED", "task was not finished after wait method"
            assert not task.running(), "task shouldn't be running after wait method returns"
            assert task.done(), "task should be 'done' after wait method"

            task = exctr.submit(calc_type="sim", dry_run=True)
            task.wait()  # should also return immediately since dry_run

            task = exctr.submit(calc_type="sim")
            job.return_value.state = "FAILED"
            task.wait(timeout=3)
            assert task.state == "FAILED", "Matching Balsam state should've been assigned to task"


@pytest.mark.extra
def test_task_kill():
    """Test of killing (cancelling) a balsam app"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    exctr = Executor.executor
    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.Job"):
        task = exctr.submit(calc_type="sim")

    with mock.patch("libensemble.executors.balsam_executors.balsam_executor.EventLog"):
        task.kill()
    assert task.finished and task.state == "USER_KILLED", "task not set as killed after kill method"


if __name__ == "__main__":
    setup_module(__file__)
    test_register_app()
    test_submit_app_defaults()
    test_submit_app_workdir()
    test_submit_app_dry()
    test_submit_app_wait()
    test_submit_revoke_alloc()
    test_task_poll()
    test_task_wait()
    test_task_kill()
    teardown_module(__file__)
