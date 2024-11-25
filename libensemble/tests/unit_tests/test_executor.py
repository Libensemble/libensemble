# !/usr/bin/env python
# Integration test of executor module for libensemble
# Test does not require running full libensemble
import os
import platform
import re
import socket
import sys
import time

import pytest

from libensemble.executors.executor import NOT_STARTED_STATES, Executor, ExecutorException, TimeoutExpired
from libensemble.message_numbers import STOP_TAG, TASK_FAILED, UNSET_TAG
from libensemble.resources.mpi_resources import MPIResourcesException

NCORES = 1
build_sims = ["my_simtask.c", "my_serialtask.c", "c_startup.c"]

sim_app = "simdir/my_simtask.x"
serial_app = "simdir/my_serialtask.x"
c_startup = "simdir/c_startup.x"
py_startup = "simdir/py_startup.py"
non_existent_app = "simdir/non_exist.x"

UNKNOWN_SIGNAL = 2000


class FakeCommTag:
    def mail_flag(self):
        return True

    def recv(self):
        return STOP_TAG + 10, 101


class FakeCommSignal:
    def mail_flag(self):
        return True

    def recv(self):
        return STOP_TAG, UNKNOWN_SIGNAL

    def push_to_buffer(self, mtag, man_signal):
        pass


def setup_module(module):
    if platform.system() != "Windows":
        import mpi4py

        mpi4py.rc.initialize = False
    try:
        print(f"setup_module module: {module.__name__}")
    except AttributeError:
        print(f"setup_module (direct run) module: {module}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None
    build_simfuncs()


def setup_function(function):
    print(f"setup_function function: {function.__name__}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def teardown_module(module):
    try:
        print(f"teardown_module module: {module.__name__}")
    except AttributeError:
        print(f"teardown_module (direct run) module: {module}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def build_simfuncs():
    import subprocess

    for sim in build_sims:
        app_name = ".".join([sim.split(".")[0], "x"])
        if not os.path.isfile(app_name):
            buildstring = "mpicc -o " + os.path.join("simdir", app_name) + " " + os.path.join("simdir", sim)
            subprocess.check_call(buildstring.split())


# This would typically be in the user calling script.
def setup_executor():
    """Set up an MPI Executor with sim app"""
    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor()
    exctr.add_platform_info()
    exctr.register_app(full_path=sim_app, calc_type="sim")


def setup_executor_no_platform():
    """Set up an MPI Executor with sim app"""
    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor()
    exctr.register_app(full_path=sim_app, calc_type="sim")


def setup_serial_executor():
    """Set up serial Executor"""
    from libensemble.executors.executor import Executor

    exctr = Executor()
    exctr.register_app(full_path=serial_app, calc_type="sim")


def setup_executor_startups():
    """Set up serial Executor"""
    from libensemble.executors.executor import Executor

    exctr = Executor()
    exctr.add_platform_info()
    exctr.register_app(full_path=c_startup, app_name="c_startup")
    exctr.register_app(full_path=py_startup, app_name="py_startup")
    exctr.register_app(full_path=py_startup, app_name="py_startup", precedent="python")


def setup_executor_noapp():
    """Set up an MPI Executor but do not register application"""
    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor()
    exctr.add_platform_info()
    if exctr.workerID is not None:
        sys.exit("Something went wrong in creating Executor")


def setup_executor_fakerunner():
    """Set up an MPI Executor with a non-existent MPI runner"""
    # Create non-existent MPI runner.
    customizer = {
        "mpi_runner": "custom",
        "runner_name": "non-existent-runner",
        "subgroup_launch": True,
    }

    from libensemble.executors.mpi_executor import MPIExecutor

    exctr = MPIExecutor(custom_info=customizer)
    exctr.add_platform_info()
    exctr.register_app(full_path=sim_app, calc_type="sim")


def is_ompi():
    """Determine if running with Open MPI"""
    import mpi4py

    mpi4py.rc.initialize = False
    from mpi4py import MPI

    return "Open MPI" in MPI.get_vendor()


# -----------------------------------------------------------------------------
# The following would typically be in the user sim_func.
def polling_loop(exctr, task, timeout_sec=2, delay=0.05):
    """Iterate over a loop, polling for an exit condition"""
    start = time.time()

    while time.time() - start < timeout_sec:
        time.sleep(delay)

        # Check output file for error.
        if task.stdout_exists():
            if "Error" in task.read_stdout() or "error" in task.read_stdout():
                print("Found(deliberate) Error in output file - cancelling task")
                exctr.kill(task)
                time.sleep(delay)  # Give time for kill
                break

        print("Polling at time", time.time() - start)
        task.poll()
        if task.finished:
            break
        elif task.state == "WAITING":
            print("Task waiting to execute")
        elif task.state == "RUNNING":
            print("Task still running ....")

    if not task.finished:
        assert task.state == "RUNNING", "task.state expected to be RUNNING. Returned: " + str(task.state)
        print("Task timed out - killing")
        exctr.kill(task)
        time.sleep(delay)  # Give time for kill
    return task


def polling_loop_multitask(exctr, task_list, timeout_sec=4.0, delay=0.05):
    """Iterate over a loop, polling for exit conditions on multiple tasks"""
    start = time.time()

    while time.time() - start < timeout_sec:
        # Test all done - (return list of not-finished tasks and test if empty)
        active_list = [task for task in task_list if not task.finished]
        if not active_list:
            break

        for task in task_list:
            if not task.finished:
                time.sleep(delay)
                print("Polling task %d at time %f" % (task.id, time.time() - start))
                task.poll()
                if task.finished:
                    continue
                elif task.state == "WAITING":
                    print("Task %d waiting to execute" % (task.id))
                elif task.state == "RUNNING":
                    print("Task %d still running ...." % (task.id))

                # Check output file for error
                if task.stdout_exists():
                    if "Error" in task.read_stdout():
                        print("Found (deliberate) Error in output file - cancelling task %d" % (task.id))
                        exctr.kill(task)
                        time.sleep(delay)  # Give time for kill
                        continue
    return task_list


# Tests ========================================================================================


def test_launch_and_poll():
    """Test of launching and polling task and exiting on task finish"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 0.2"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    assert task.run_attempts == 1, "task.run_attempts should be 1. Returned " + str(task.run_attempts)


def test_launch_and_wait():
    """Test of launching and waiting on task"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 1"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task.wait()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    task.wait()  # Already complete
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    err_code = task.exception()
    assert err_code == 0, f"Expected error code 0. Returned {err_code}"


def test_launch_and_wait_no_platform():
    """Test of launching and waiting on task with no platform setup

    The MPI runner should be set on first call to executor.
    """
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor_no_platform()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 1"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task.wait()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    task.wait()  # Already complete
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)


def test_launch_and_wait_timeout():
    """Test of launching and waiting on task timeout (and kill)"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 5"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    try:
        task.wait(timeout=0.5)
    except TimeoutExpired as e:
        print(task)
        print(e)
        assert not task.finished, "task.finished should be False. Returned " + str(task.finished)
        task.kill()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "USER_KILLED", "task.state should be USER_KILLED. Returned " + str(task.state)


def test_launch_wait_on_start():
    """Test of launching task with wait_on_start"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 0.2"
    for value in [False, True]:
        task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim, wait_on_start=value)
        assert task.state not in NOT_STARTED_STATES, "Task should not be in a NOT_STARTED state. State: " + str(
            task.state
        )
        exctr.poll(task)
        if not task.finished:
            task = polling_loop(exctr, task)
        assert task.finished, "task.finished should be True. Returned " + str(task.finished)
        assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)


def test_kill_on_file():
    """Test of killing task based on something in output file"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 0.2 Error"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "USER_KILLED", "task.state should be USER_KILLED. Returned " + str(task.state)


def test_kill_on_timeout():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 10"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "USER_KILLED", "task.state should be USER_KILLED. Returned " + str(task.state)


def test_kill_on_timeout_polling_loop_method():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 10"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    exctr.polling_loop(task, timeout=1)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "USER_KILLED", "task.state should be USER_KILLED. Returned " + str(task.state)


def test_launch_and_poll_multitasks():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    task_list = []
    cores = NCORES

    for j in range(3):
        # outfilename = 'out_' + str(j) + '.txt' # Could allow launch to generate outfile names based on task.id
        outfile = "multitask_task_" + str(j) + ".out"
        sleeptime = 0.3 + (j * 0.2)  # Change args
        args_for_sim = "sleep" + " " + str(sleeptime)
        # rundir = 'run_' + str(sleeptime)
        task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim, stdout=outfile)
        task_list.append(task)

    task_list_return = polling_loop_multitask(exctr, task_list)
    for task in task_list_return:
        assert task.finished, "task.finished should be True. Returned " + str(task.finished)
        assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)


def test_get_task():
    """Return task from given task id"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor

    # Try with no tasks set up
    A = exctr.get_task("a")
    assert A is None, "Task found when tasklist should be empty"

    # Set up task and getid
    cores = NCORES
    args_for_sim = "sleep 0"
    task0 = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    taskid = task0.id
    print(f"taskid is: {taskid}")
    A = exctr.get_task(taskid)
    assert A is task0, "Task get_task returned unexpected task" + str(A)
    task0 = polling_loop(exctr, task0)

    # Get non-existent taskid
    A = exctr.get_task(taskid + 1)
    assert A is None, "Task found when supplied taskid should not exist"


@pytest.mark.timeout(30)
def test_procs_and_machinefile_logic():
    """Test of supplying various input configurations."""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    # Note: Could test task_partition routine directly - without launching tasks...

    # Testing machinefile
    setup_executor()
    exctr = Executor.executor
    args_for_sim = "sleep 0"

    machinefilename = "my_machinefile"
    cores = NCORES
    with open(machinefilename, "w") as f:
        for rank in range(cores):
            f.write(socket.gethostname() + "\n")

    task = exctr.submit(calc_type="sim", machinefile=machinefilename, app_args=args_for_sim)
    task = polling_loop(exctr, task, timeout_sec=4, delay=0.05)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)

    # Testing num_procs = num_nodes*procs_per_node (shouldn't fail)
    if is_ompi():
        task = exctr.submit(
            calc_type="sim",
            num_procs=6,
            num_nodes=1,
            procs_per_node=6,
            app_args=args_for_sim,
            extra_args="--oversubscribe",
        )
    else:
        task = exctr.submit(calc_type="sim", num_procs=6, num_nodes=2, procs_per_node=3, app_args=args_for_sim)
    task = polling_loop(exctr, task, timeout_sec=4, delay=0.05)
    time.sleep(0.25)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)

    # Testing num_procs not num_nodes*procs_per_node (should fail).
    try:
        task = exctr.submit(calc_type="sim", num_procs=9, num_nodes=2, procs_per_node=5, app_args=args_for_sim)
    except MPIResourcesException as e:
        assert e.args[0] == "num_procs does not equal num_nodes*procs_per_node"
    else:
        assert 0

    # Testing no num_procs (should not fail).
    if is_ompi():
        task = exctr.submit(
            calc_type="sim",
            num_nodes=1,
            procs_per_node=3,
            app_args=args_for_sim,
            extra_args="--oversubscribe",
        )
    else:
        task = exctr.submit(calc_type="sim", num_nodes=2, procs_per_node=3, app_args=args_for_sim)
    assert 1
    task = polling_loop(exctr, task, timeout_sec=4, delay=0.05)
    time.sleep(0.25)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)

    # Testing nothing given (should fail).
    try:
        task = exctr.submit(calc_type="sim", app_args=args_for_sim)
    except MPIResourcesException as e:
        assert e.args[0] == "Need num_procs, num_nodes/procs_per_node, or machinefile"
    else:
        assert 0

    # Testing no num_nodes (should not fail).
    task = exctr.submit(calc_type="sim", num_procs=2, procs_per_node=2, app_args=args_for_sim)
    assert 1
    task = polling_loop(exctr, task, timeout_sec=4, delay=0.05)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)

    # Testing no procs_per_node (shouldn't fail)
    task = exctr.submit(calc_type="sim", num_nodes=1, num_procs=2, app_args=args_for_sim)
    assert 1
    task = polling_loop(exctr, task, timeout_sec=4, delay=0.05)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)

    # Test with jsrun - does not support machinefiles
    with pytest.raises(MPIResourcesException):
        task = exctr.submit(
            calc_type="sim", machinefile=machinefilename, app_args=args_for_sim, mpi_runner_type="jsrun"
        )


@pytest.mark.timeout(20)
def test_doublekill():
    """Test attempt to kill already killed task.

    Kill should have no effect (except warning message) and should remain in state killed.
    """
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 2.0"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task.poll()
    exctr.wait_time = 5

    exctr.kill(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "USER_KILLED", "task.state should be USER_KILLED. Returned " + str(task.state)
    exctr.kill(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "USER_KILLED", "task.state should be USER_KILLED. Returned " + str(task.state)


@pytest.mark.timeout(20)
def test_finish_and_kill():
    """Test attempt to kill already finished task.

    Kill should have no effect (except warning message) and should remain in state FINISHED.
    """
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 0.1"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    while not task.finished:
        time.sleep(0.1)
        task.poll()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    exctr.kill(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    # Try polling after finish - should return with no effect
    task.poll()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)


@pytest.mark.timeout(20)
def test_launch_and_kill():
    """Test launching and immediately killing tasks with no poll"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 2.0"
    task_list = []
    exctr.wait_time = 1
    for taskid in range(5):
        task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
        exctr.kill(task)
        task_list.append(task)

    for task in task_list:
        assert task.finished, "task.finished should be True. Returned " + str(task.finished)
        assert task.state == "USER_KILLED", "task.state should be USER_KILLED. Returned " + str(task.state)


def test_launch_as_gen():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 0.1"

    # Try launching as gen when not registered as gen.
    try:
        task = exctr.submit(calc_type="gen", num_procs=cores, app_args=args_for_sim)
    except ExecutorException as e:
        assert e.args[0] == "Default gen app is not set"
    else:
        assert 0

    exctr.register_app(full_path=sim_app, calc_type="gen")
    task = exctr.submit(calc_type="gen", num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)

    # Try launching as 'alloc', which is not a type.
    try:
        task = exctr.submit(calc_type="alloc", num_procs=cores, app_args=args_for_sim)
    except ExecutorException as e:
        assert e.args[0] + e.args[1] == "Unrecognized calculation type" + "alloc"
    else:
        assert 0


def test_launch_no_app():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor_noapp()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 0.1"
    try:
        _ = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    except ExecutorException as e:
        assert e.args[0] == "Default sim app is not set"
    else:
        assert 0
    try:
        _ = exctr.submit(num_procs=cores, app_args=args_for_sim)
    except ExecutorException as e:
        assert e.args[0] == "Either app_name or calc_type must be set"
    else:
        assert 0


def test_kill_task_with_no_submit():
    from libensemble.executors.executor import Task

    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor

    # Try kill invalid task
    try:
        exctr.kill("mytask")
    except ExecutorException as e:
        assert e.args[0] == "Invalid task has been provided"
    else:
        assert 0

    # Create a task directly with no submit (not supported for users).
    # Debatably make taskID 0 as executor should be deleted if use setup function.
    # But this allows any task ID.
    exp_msg = "task libe_task_my_simtask.x_.+has no process ID - check task has been launched"
    exp_re = re.compile(exp_msg)
    myapp = exctr.sim_default_app
    task1 = Task(app=myapp, stdout="stdout.txt")
    try:
        exctr.kill(task1)
    except ExecutorException as e:
        assert bool(re.match(exp_re, e.args[0]))
    else:
        assert 0


def test_poll_task_with_no_submit():
    from libensemble.executors.executor import Task

    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor

    # Create a task directly with no submit (Not supported for users).
    exp_msg = "task libe_task_my_simtask.x_.+has no process ID - check task has been launched"
    exp_re = re.compile(exp_msg)
    myapp = exctr.sim_default_app
    task1 = Task(app=myapp, stdout="stdout.txt")
    try:
        task1.poll()
    except ExecutorException as e:
        assert bool(re.match(exp_re, e.args[0]))
    else:
        assert 0


def test_task_failure():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 1.0 Fail"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task, timeout_sec=3)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FAILED", "task.state should be FAILED. Returned " + str(task.state)


def test_task_failure_polling_loop_method():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 1.0 Fail"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    calc_status = exctr.polling_loop(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FAILED", "task.state should be FAILED. Returned " + str(task.state)
    assert calc_status == TASK_FAILED, f"calc_status should be {TASK_FAILED}"


def test_task_unknown_state():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 1.0"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim, dry_run=True)
    task.state = "unknown"
    calc_status = exctr.polling_loop(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert calc_status == UNSET_TAG, f"calc_status should be {UNSET_TAG}. Found {calc_status}"


def test_retries_launch_fail():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor_fakerunner()
    exctr = Executor.executor
    exctr.retry_delay_incr = 0.05
    cores = NCORES
    args_for_sim = "sleep 0"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    assert task.state == "FAILED_TO_START", "task.state should be FAILED_TO_START. Returned " + str(task.state)
    assert exctr.mpi_runner_obj.subgroup_launch, "subgroup_launch should be True"
    assert task.run_attempts == 5, "task.run_attempts should be 5. Returned " + str(task.run_attempts)


def test_retries_before_polling_loop_method():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor_fakerunner()
    exctr = Executor.executor
    exctr.retry_delay_incr = 0.05
    cores = NCORES
    args_for_sim = "sleep 0"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim)
    exctr.polling_loop(task, timeout=1)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FAILED_TO_START", "task.state should be FAILED_TO_START. Returned " + str(task.state)
    assert task.run_attempts == 5, "task.run_attempts should be 5. Returned " + str(task.run_attempts)


def test_retries_run_fail():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    exctr.retry_delay_incr = 0.05
    cores = NCORES
    args_for_sim = "sleep 0 Fail"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim, wait_on_start=True)
    assert task.state == "FAILED", "task.state should be FAILED. Returned " + str(task.state)
    assert task.run_attempts == 5, "task.run_attempts should be 5. Returned " + str(task.run_attempts)


def test_register_apps():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()  # This registers an app my_simtask.x (default sim)
    exctr = Executor.executor
    exctr.register_app(full_path="/path/to/fake_app1.x", app_name="fake_app1")
    exctr.register_app(full_path="/path/to/fake_app2.py", app_name="fake_app2")
    exctr.register_app(full_path="/path/to/fake_app3.pl", app_name="fake_app3", precedent="perl")

    # Check selected attributes
    app = exctr.get_app("my_simtask.x")
    assert app.name == "my_simtask.x"
    assert app.gname == "libe_app_my_simtask.x"

    app = exctr.get_app("fake_app1")
    assert app.name == "fake_app1"
    assert app.gname == "libe_app_fake_app1"
    assert app.exe == "fake_app1.x"
    assert app.calc_dir == "/path/to"
    assert app.app_cmd == "/path/to/fake_app1.x"
    assert not app.precedent

    app = exctr.get_app("fake_app2")
    assert app.name == "fake_app2"
    assert app.gname == "libe_app_fake_app2"
    assert app.full_path == "/path/to/fake_app2.py"

    py_exe, app_exe = app.app_cmd.split()
    assert os.path.split(py_exe)[1].startswith("python")
    assert app_exe == "/path/to/fake_app2.py"

    app = exctr.get_app("fake_app3")
    assert app.name == "fake_app3"
    assert app.gname == "libe_app_fake_app3"
    assert app.full_path == "/path/to/fake_app3.pl"
    assert app.precedent == "perl"
    assert app.app_cmd == "perl /path/to/fake_app3.pl"

    try:
        app = exctr.get_app("fake_app4")
    except ExecutorException as e:
        assert e.args[0] == "Application fake_app4 not found in registry"
        # Ordering of dictionary may vary
        # assert e.args[1] == "Registered applications: ['my_simtask.x', 'fake_app1', 'fake_app2']"


def test_serial_exes():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_serial_executor()
    exctr = Executor.executor
    args_for_sim = "sleep 0.1"
    task = exctr.submit(calc_type="sim", app_args=args_for_sim, wait_on_start=True)
    task.wait()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)


def test_serial_exe_exception():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_serial_executor()
    exctr = Executor.executor
    with pytest.raises(ExecutorException):
        exctr.submit()
        pytest.fail("Expected exception")


def test_serial_exe_env_script():
    env_script_path = os.path.join(os.getcwd(), "./env_script_in.sh")
    setup_serial_executor()
    exctr = Executor.executor
    args_for_sim = "sleep 0.1"
    task = exctr.submit(calc_type="sim", app_args=args_for_sim, env_script=env_script_path)
    task.wait()
    # test env var should not exist here
    assert "LIBE_TEST_SUB_ENV_VAR" not in os.environ


def test_serial_exe_dryrun():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_serial_executor()
    exctr = Executor.executor
    exctr.set_gen_procs_gpus(libE_info={})
    exctr.set_workerID(1)
    args_for_sim = "sleep 0.1"
    task = exctr.submit(calc_type="sim", app_args=args_for_sim, dry_run=True)
    task.wait()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)


def test_serial_startup_times():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor_startups()
    exctr = Executor.executor

    t1 = time.time()
    task = exctr.submit(app_name="c_startup")
    task.wait()
    stime = float(task.read_stdout())
    startup_time = stime - t1
    print("start up time for c program", startup_time)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    assert 0 < startup_time < 1, "Start up time for C program took " + str(startup_time)

    t1 = time.time()
    task = exctr.submit(app_name="py_startup")
    task.wait()
    stime = float(task.read_stdout())
    startup_time = stime - t1
    print("start up time for python program", startup_time)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == "FINISHED", "task.state should be FINISHED. Returned " + str(task.state)
    assert 0 < startup_time < 1, "Start up time for python program took " + str(startup_time)


def test_futures_interface():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    cores = NCORES
    args_for_sim = "sleep 3"
    with Executor.executor as exctr:
        task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim, wait_on_start=True)
    time.sleep(0.1)
    assert task.running(), "task.running() should return True after wait_on_start task submission."
    assert task.result() == "FINISHED", "task.result() should return FINISHED. Returned " + str(task.state)
    assert task.done(), "task.done() should return True after task finishes."


def test_futures_interface_cancel():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    cores = NCORES
    args_for_sim = "sleep 3"
    with Executor.executor as exctr:
        task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim, wait_on_start=True)
    time.sleep(0.1)
    task.cancel()
    assert task.cancelled() and task.done(), "Task should be both cancelled() and done() after cancellation."


def test_dry_run():
    """Test of dry_run in poll"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = "sleep 0.2"
    task = exctr.submit(calc_type="sim", num_procs=cores, app_args=args_for_sim, dry_run=True)
    task.poll()
    task.kill()


def test_non_existent_app():
    """Tests exception on non-existent app"""
    from libensemble.executors.executor import Executor

    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    exctr = Executor()

    # Can register a non-existent app in case created as part of workflow.
    exctr.register_app(full_path=non_existent_app, app_name="nonexist")

    w_exctr = Executor.executor  # simulate on worker

    try:
        w_exctr.submit(app_name="nonexist")
    except ExecutorException as e:
        assert e.args[0] == "Application does not exist simdir/non_exist.x"
    else:
        assert 0


def test_non_existent_app_mpi():
    """Tests exception on non-existent app"""
    from libensemble.executors.mpi_executor import MPIExecutor

    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    exctr = MPIExecutor()

    # Can register a non-existent app in case created as part of workflow.
    exctr.register_app(full_path=non_existent_app, app_name="nonexist")

    w_exctr = Executor.executor  # simulate on worker

    try:
        w_exctr.submit(app_name="nonexist")
    except ExecutorException as e:
        assert e.args[0] == "Application does not exist simdir/non_exist.x"
    else:
        assert 0


def test_man_signal_unrec_tag():
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    setup_serial_executor()
    exctr = Executor.executor

    fake_comm1 = FakeCommTag()
    exctr.comm = fake_comm1
    man_signal = exctr.manager_poll()
    assert man_signal is None, "manager_poll should have returned None"

    fake_comm2 = FakeCommSignal()
    exctr.comm = fake_comm2
    man_signal = exctr.manager_poll()
    assert man_signal == UNKNOWN_SIGNAL, f"manager_poll should have returned {UNKNOWN_SIGNAL}. Received {man_signal}"


if __name__ == "__main__":
    setup_module(__file__)
    test_launch_and_poll()
    test_launch_and_wait()
    test_launch_and_wait_no_platform()
    test_launch_and_wait_timeout()
    test_launch_wait_on_start()
    test_kill_on_file()
    test_kill_on_timeout()
    test_kill_on_timeout_polling_loop_method()
    test_launch_and_poll_multitasks()
    test_get_task()
    test_procs_and_machinefile_logic()
    test_doublekill()
    test_finish_and_kill()
    test_launch_and_kill()
    test_launch_as_gen()
    test_launch_no_app()
    test_kill_task_with_no_submit()
    test_poll_task_with_no_submit()
    test_task_failure()
    test_task_failure_polling_loop_method()
    test_task_unknown_state()
    test_retries_launch_fail()
    test_retries_before_polling_loop_method()
    test_retries_run_fail()
    test_register_apps()
    test_serial_exes()
    test_serial_exe_exception()
    test_serial_exe_env_script()
    test_serial_exe_dryrun()
    test_serial_startup_times()
    test_futures_interface()
    test_futures_interface_cancel()
    test_dry_run()
    test_non_existent_app()
    test_non_existent_app_mpi()
    test_man_signal_unrec_tag()
    teardown_module(__file__)
