# !/usr/bin/env python
# Integration Test of executor module for libensemble
# Test does not require running full libensemble
import os
import sys
import time
import pytest
import socket
from libensemble.resources.resources import ResourcesException
from libensemble.executors.executor import Executor, ExecutorException
from libensemble.executors.executor import NOT_STARTED_STATES


USE_BALSAM = False

NCORES = 1
sim_app = './my_simtask.x'


def setup_module(module):
    print("setup_module      module:%s" % module.__name__)
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def setup_function(function):
    print("setup_function    function:%s" % function.__name__)
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def teardown_module(module):
    print("teardown_module   module:%s" % module.__name__)
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simtask.x my_simtask.f90' # On cray need to use ftn
    buildstring = 'mpicc -o my_simtask.x simdir/my_simtask.c'
    # subprocess.run(buildstring.split(), check=True) # Python3.5+
    subprocess.check_call(buildstring.split())


# This would typically be in the user calling script
# Cannot test auto_resources here - as no workers set up.
def setup_executor():
    # sim_app = './my_simtask.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        from libensemble.executors.balsam_executor import BalsamMPIExecutor
        exctr = BalsamMPIExecutor(auto_resources=False)
    else:
        from libensemble.executors.mpi_executor import MPIExecutor
        exctr = MPIExecutor(auto_resources=False)

    exctr.register_calc(full_path=sim_app, calc_type='sim')


def setup_executor_noreg():
    # sim_app = './my_simtask.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        from libensemble.executors.balsam_executor import BalsamMPIExecutor
        exctr = BalsamMPIExecutor(auto_resources=False)
    else:
        from libensemble.executors.mpi_executor import MPIExecutor
        exctr = MPIExecutor(auto_resources=False)

    exctr.register_calc(full_path=sim_app, calc_type='sim')


def setup_executor_noapp():
    # sim_app = './my_simtask.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        from libensemble.executors.balsam_executor import BalsamMPIExecutor
        exctr = BalsamMPIExecutor(auto_resources=False)
    else:
        from libensemble.executors.mpi_executor import MPIExecutor
        exctr = MPIExecutor(auto_resources=False)
        if exctr.workerID is not None:
            sys.exit("Something went wrong in creating Executor")


def setup_executor_fakerunner():
    # sim_app = './my_simtask.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        print('Balsom does not support this feature - running MPIExecutor')

    # Create non-existent MPI runner.
    customizer = {'mpi_runner': 'custom',
                  'runner_name': 'non-existent-runner',
                  'subgroup_launch': True}

    from libensemble.executors.mpi_executor import MPIExecutor
    exctr = MPIExecutor(auto_resources=False, custom_info=customizer)
    exctr.register_calc(full_path=sim_app, calc_type='sim')


# -----------------------------------------------------------------------------
# The following would typically be in the user sim_func
def polling_loop(exctr, task, timeout_sec=0.5, delay=0.05):
    # import time
    start = time.time()

    while time.time() - start < timeout_sec:
        time.sleep(delay)

        # Check output file for error
        if task.stdout_exists():
            if 'Error' in task.read_stdout():
                print("Found(deliberate) Error in ouput file - cancelling task")
                exctr.kill(task)
                time.sleep(delay)  # Give time for kill
                break

        print('Polling at time', time.time() - start)
        task.poll()
        if task.finished:
            break
        elif task.state == 'WAITING':
            print('Task waiting to execute')
        elif task.state == 'RUNNING':
            print('Task still running ....')

    if not task.finished:
        assert task.state == 'RUNNING', "task.state expected to be RUNNING. Returned: " + str(task.state)
        print("Task timed out - killing")
        exctr.kill(task)
        time.sleep(delay)  # Give time for kill
    return task


def polling_loop_multitask(exctr, task_list, timeout_sec=4.0, delay=0.05):
    # import time
    start = time.time()

    while time.time() - start < timeout_sec:

        # Test all done - (return list of not-finished tasks and test if empty)
        active_list = [task for task in task_list if not task.finished]
        if not active_list:
            break

        for task in task_list:
            if not task.finished:
                time.sleep(delay)
                print('Polling task %d at time %f' % (task.id, time.time() - start))
                task.poll()
                if task.finished:
                    continue
                elif task.state == 'WAITING':
                    print('Task %d waiting to execute' % (task.id))
                elif task.state == 'RUNNING':
                    print('Task %d still running ....' % (task.id))

                # Check output file for error
                if task.stdout_exists():
                    if 'Error' in task.read_stdout():
                        print("Found (deliberate) Error in ouput file - cancelling task %d" % (task.id))
                        exctr.kill(task)
                        time.sleep(delay)  # Give time for kill
                        continue
    return task_list


# Tests ========================================================================================
def test_launch_and_poll():
    """ Test of launching and polling task and exiting on task finish"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 0.2'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)
    assert task.run_attempts == 1, "task.run_attempts should be 1. Returned " + str(task.run_attempts)


def test_launch_wait_on_run():
    """ Test of launching task with wait_on_run """
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 0.2'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim, wait_on_run=True)
    assert task.state not in NOT_STARTED_STATES, "Task should not be in a NOT_STARTED state. State: " + str(task.state)
    exctr.poll(task)
    if not task.finished:
        task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)


def test_kill_on_file():
    """ Test of killing task based on something in output file"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 0.1 Error'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'USER_KILLED', "task.state should be USER_KILLED. Returned " + str(task.state)


def test_kill_on_timeout():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 10'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'USER_KILLED', "task.state should be USER_KILLED. Returned " + str(task.state)


def test_launch_and_poll_multitasks():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    task_list = []
    cores = NCORES

    for j in range(3):
        # outfilename = 'out_' + str(j) + '.txt' # Could allow launch to generate outfile names based on task.id
        outfile = 'multitask_task_' + str(j) + '.out'
        sleeptime = 0.3 + (j*0.2)  # Change args
        args_for_sim = 'sleep' + ' ' + str(sleeptime)
        # rundir = 'run_' + str(sleeptime)
        task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim, stdout=outfile)
        task_list.append(task)

    task_list_return = polling_loop_multitask(exctr, task_list)
    for task in task_list_return:
        assert task.finished, "task.finished should be True. Returned " + str(task.finished)
        assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)


def test_get_task():
    """Return task from given task id"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor

    # Try with no tasks set up
    A = exctr.get_task('a')
    assert A is None, 'Task found when tasklist should be empty'

    # Set up task and getid
    cores = NCORES
    args_for_sim = 'sleep 0'
    task0 = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    taskid = task0.id
    print("taskid is: {}".format(taskid))
    A = exctr.get_task(taskid)
    assert A is task0, 'Task get_task returned unexpected task' + str(A)
    task0 = polling_loop(exctr, task0)

    # Get non-existent taskid
    A = exctr.get_task(taskid+1)
    assert A is None, 'Task found when supplied taskid should not exist'


@pytest.mark.timeout(30)
def test_procs_and_machinefile_logic():
    """ Test of supplying various input configurations when auto_resources is False."""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))

    # Note: Could test task_partition routine directly - without launching tasks...

    # Testing machinefile
    setup_executor()
    exctr = Executor.executor
    args_for_sim = 'sleep 0'

    machinefilename = "my_machinefile"
    cores = NCORES
    with open(machinefilename, 'w') as f:
        for rank in range(cores):
            f.write(socket.gethostname() + '\n')

    task = exctr.submit(calc_type='sim', machinefile=machinefilename, app_args=args_for_sim)
    task = polling_loop(exctr, task, delay=0.02)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)

    # Testing num_procs = num_nodes*ranks_per_node (shouldn't fail)
    task = exctr.submit(calc_type='sim', num_procs=6, num_nodes=2, ranks_per_node=3, app_args=args_for_sim)
    task = polling_loop(exctr, task, delay=0.02)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)

    # Testing num_procs not num_nodes*ranks_per_node (should fail)
    try:
        task = exctr.submit(calc_type='sim', num_procs=9, num_nodes=2, ranks_per_node=5, app_args=args_for_sim)
    except ResourcesException as e:
        assert e.args[0] == 'num_procs does not equal num_nodes*ranks_per_node'
    else:
        assert 0

    # Testing no num_procs (shouldn't fail)
    task = exctr.submit(calc_type='sim', num_nodes=2, ranks_per_node=3, app_args=args_for_sim)
    assert 1
    task = polling_loop(exctr, task, delay=0.02)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)

    # Testing nothing given (should fail)
    try:
        task = exctr.submit(calc_type='sim', app_args=args_for_sim)
    except ResourcesException as e:
        assert e.args[0] == 'Need num_procs, num_nodes/ranks_per_node, or machinefile'
    else:
        assert 0

    # Testing no num_nodes (shouldn't fail)
    task = exctr.submit(calc_type='sim', num_procs=2, ranks_per_node=2, app_args=args_for_sim)
    assert 1
    task = polling_loop(exctr, task, delay=0.02)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)

    # Testing no ranks_per_node (shouldn't fail)
    task = exctr.submit(calc_type='sim', num_nodes=1, num_procs=2, app_args=args_for_sim)
    assert 1
    task = polling_loop(exctr, task, delay=0.02)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)


@pytest.mark.timeout(20)
def test_doublekill():
    """Test attempt to kill already killed task

    Kill should have no effect (except warning message) and should remain in state killed
    """
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 2.0'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    task.poll()
    exctr.wait_time = 5

    exctr.kill(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'USER_KILLED', "task.state should be USER_KILLED. Returned " + str(task.state)
    exctr.kill(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'USER_KILLED', "task.state should be USER_KILLED. Returned " + str(task.state)


@pytest.mark.timeout(20)
def test_finish_and_kill():
    """Test attempt to kill already finished task

    Kill should have no effect (except warning message) and should remain in state FINISHED
    """
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    while not task.finished:
        time.sleep(0.1)
        task.poll()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)
    exctr.kill(task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)
    # Try polling after finish - should return with no effect
    task.poll()
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)


@pytest.mark.timeout(20)
def test_launch_and_kill():
    """Test launching and immediately killing tasks with no poll"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 2.0'
    task_list = []
    exctr.wait_time = 1
    for taskid in range(5):
        task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
        exctr.kill(task)
        task_list.append(task)

    for task in task_list:
        assert task.finished, "task.finished should be True. Returned " + str(task.finished)
        assert task.state == 'USER_KILLED', "task.state should be USER_KILLED. Returned " + str(task.state)


def test_launch_as_gen():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 0.1'

    # Try launching as gen when not registered as gen
    try:
        task = exctr.submit(calc_type='gen', num_procs=cores, app_args=args_for_sim)
    except ExecutorException as e:
        assert e.args[0] == 'Default gen app is not set'
    else:
        assert 0

    exctr.register_calc(full_path=sim_app, calc_type='gen')
    task = exctr.submit(calc_type='gen', num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)

    # Try launching as 'alloc' which is not a type
    try:
        task = exctr.submit(calc_type='alloc', num_procs=cores, app_args=args_for_sim)
    except ExecutorException as e:
        assert e.args[0] + e.args[1] == 'Unrecognized calculation type' + 'alloc'
    else:
        assert 0


def test_launch_default_reg():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor_noreg()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FINISHED', "task.state should be FINISHED. Returned " + str(task.state)


def test_launch_no_app():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor_noapp()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    try:
        _ = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    except ExecutorException as e:
        assert e.args[0] == 'Default sim app is not set'
    else:
        assert 0


def test_kill_task_with_no_submit():
    from libensemble.executors.executor import Task
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor

    # Try kill invalid task
    try:
        exctr.kill('mytask')
    except ExecutorException as e:
        assert e.args[0] == 'Invalid task has been provided'
    else:
        assert 0

    # Create a task directly with no submit (Not supported for users)
    myapp = exctr.sim_default_app
    task1 = Task(app=myapp, stdout='stdout.txt')
    try:
        exctr.kill(task1)
    except ExecutorException as e:
        assert e.args[0][:50] == 'Attempting to kill task task_my_simtask.x.simfunc_'
        assert e.args[0][52:] == ' that has no process ID - check tasks been launched'
    else:
        assert 0


def test_poll_task_with_no_submit():
    from libensemble.executors.executor import Task
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor

    # Create a task directly with no submit (Not supported for users)
    myapp = exctr.sim_default_app
    task1 = Task(app=myapp, stdout='stdout.txt')
    try:
        task1.poll()
    except ExecutorException as e:
        assert e.args[0][:38] == 'Polled task task_my_simtask.x.simfunc_'
        assert e.args[0][40:] == ' has no process ID - check tasks been launched'
    else:
        assert 0


def test_task_failure():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    cores = NCORES
    args_for_sim = 'sleep 1.0 Fail'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    task = polling_loop(exctr, task)
    assert task.finished, "task.finished should be True. Returned " + str(task.finished)
    assert task.state == 'FAILED', "task.state should be FAILED. Returned " + str(task.state)


def test_retries_launch_fail():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor_fakerunner()
    exctr = Executor.executor
    exctr.retry_delay_incr = 0.05
    cores = NCORES
    args_for_sim = 'sleep 0'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    assert task.state == 'CREATED', "task.state should be CREATED. Returned " + str(task.state)
    assert exctr.mpi_runner.subgroup_launch, "subgroup_launch should be True"
    assert task.run_attempts == 5, "task.run_attempts should be 5. Returned " + str(task.run_attempts)


def test_retries_run_fail():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_executor()
    exctr = Executor.executor
    exctr.retry_delay_incr = 0.05
    cores = NCORES
    args_for_sim = 'sleep 0 Fail'
    task = exctr.submit(calc_type='sim', num_procs=cores, app_args=args_for_sim, wait_on_run=True)
    assert task.state == 'FAILED', "task.state should be FAILED. Returned " + str(task.state)
    assert task.run_attempts == 5, "task.run_attempts should be 5. Returned " + str(task.run_attempts)


if __name__ == "__main__":
    # setup_module(__file__)
    test_launch_and_poll()
    test_launch_wait_on_run()
    test_kill_on_file()
    test_kill_on_timeout()
    test_launch_and_poll_multitasks()
    test_get_task()
    test_procs_and_machinefile_logic()
    test_doublekill()
    test_finish_and_kill()
    test_launch_and_kill()
    test_launch_as_gen()
    test_launch_default_reg()
    test_launch_no_app()
    test_kill_task_with_no_submit()
    test_poll_task_with_no_submit()
    test_task_failure()
    test_retries_launch_fail()
    test_retries_run_fail()
    # teardown_module(__file__)
