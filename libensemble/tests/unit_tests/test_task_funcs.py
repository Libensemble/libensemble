import os
import shutil
import time

from libensemble.executors.executor import Task, Executor, ExecutorException
from libensemble.executors.mpi_executor import MPIExecutor


def setup_module(module):
    print(f"setup_module      module:{module.__name__}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def setup_function(function):
    print(f"setup_function    function:{function.__name__}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def teardown_module(module):
    print(f"teardown_module   module:{module.__name__}")
    if Executor.executor is not None:
        del Executor.executor
        Executor.executor = None


def test_task_funcs():
    dummyappname = os.getcwd() + "/myapp.x"
    exctr = MPIExecutor()
    exctr.register_app(full_path=dummyappname, calc_type="gen", desc="A dummy calc")
    exctr.register_app(full_path=dummyappname, calc_type="sim", desc="A dummy calc")

    dirname = "dir_taskc_tests"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.chdir(dirname)
    myworkdir = os.getcwd()

    # First try no app - check exception raised?
    jc_triggered = False
    try:
        _ = Task(workdir=myworkdir, stdout="stdout.txt", stderr="stderr.txt")
    except ExecutorException:
        jc_triggered = True

    assert jc_triggered, "Failed to raise exception if create task with no app"

    # Now with no workdir specified
    dummyapp = exctr.gen_default_app
    task1 = Task(app=dummyapp, stdout="stdout.txt", stderr="stderr.txt")
    wd_exist = task1.workdir_exists()
    assert not wd_exist  # , "No workdir specified, yet workdir_exists does not return False"
    stdout_exist = task1.stdout_exists()
    assert not stdout_exist
    f_exist = task1.file_exists_in_workdir("running_output.txt")
    assert not f_exist

    # Create task properly specified
    task2 = Task(app=dummyapp, workdir=myworkdir, stdout="stdout.txt", stderr="stderr.txt")

    # Workdir does exist
    wd_exist = task2.workdir_exists()
    assert wd_exist

    # Files do not exist
    stdout_exist = task2.stdout_exists()
    assert not stdout_exist
    stderr_exist = task2.stderr_exists()
    assert not stderr_exist
    f_exist = task2.file_exists_in_workdir("running_output.txt")
    assert not f_exist

    valerr_triggered = False
    try:
        task2.read_stdout()
    except ValueError:
        valerr_triggered = True
    assert valerr_triggered

    valerr_triggered = False
    try:
        task2.read_file_in_workdir("running_output.txt")
    except ValueError:
        valerr_triggered = True
    assert valerr_triggered

    # Now create files and check positive results
    with open("stdout.txt", "w") as f:
        f.write("This is stdout")
    with open("stderr.txt", "w") as f:
        f.write("This is stderr")
    with open("running_output.txt", "w") as f:
        f.write("This is running output")

    # task2 = Task(app = dummyapp, workdir = myworkdir, stdout = 'stdout.txt')
    # wd_exist = task2.workdir_exists()
    # assert wd_exist
    stdout_exist = task2.stdout_exists()
    assert stdout_exist
    stderr_exist = task2.stderr_exists()
    assert stderr_exist
    f_exist = task2.file_exists_in_workdir("running_output.txt")
    assert f_exist
    assert "This is stdout" in task2.read_stdout()
    assert "This is stderr" in task2.read_stderr()
    assert "This is running output" in task2.read_file_in_workdir("running_output.txt")

    # Check if workdir does not exist
    task2.workdir = task2.workdir + "/bubbles"
    wd_exist = task2.workdir_exists()
    assert not wd_exist

    # Check timing
    assert not task2.submit_time and not task2.runtime and not task2.total_time
    task2.calc_task_timing()
    assert not task2.submit_time and not task2.runtime and not task2.total_time
    task2.submit_time = time.time()
    task2.calc_task_timing()
    assert task2.runtime is not None and task2.runtime == task2.total_time
    save_runtime, save_total_time = task2.runtime, task2.total_time
    task2.calc_task_timing()
    assert save_runtime == task2.runtime
    assert save_total_time == task2.total_time

    # Clean up
    os.chdir("../")
    shutil.rmtree(dirname)


if __name__ == "__main__":
    test_task_funcs()
