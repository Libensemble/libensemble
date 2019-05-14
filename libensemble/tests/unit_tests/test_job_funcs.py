import os
import shutil
import time

from libensemble.controller import Job, JobController, JobControllerException
from libensemble.mpi_controller import MPIJobController


def setup_module(module):
    print("setup_module      module:%s" % module.__name__)
    if JobController.controller is not None:
        del JobController.controller
        JobController.controller = None


def setup_function(function):
    print("setup_function    function:%s" % function.__name__)
    if JobController.controller is not None:
        del JobController.controller
        JobController.controller = None


def teardown_module(module):
    print("teardown_module   module:%s" % module.__name__)
    if JobController.controller is not None:
        del JobController.controller
        JobController.controller = None


def test_job_funcs():
    dummyappname = os.getcwd() + '/myapp.x'
    jobctrl = MPIJobController(auto_resources=False)
    jobctrl.register_calc(full_path=dummyappname, calc_type='gen', desc='A dummy calc')
    jobctrl.register_calc(full_path=dummyappname, calc_type='sim', desc='A dummy calc')

    dirname = 'dir_jobc_tests'
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.chdir(dirname)
    myworkdir = os.getcwd()

    # First try no app - check exception raised?
    jc_triggered = False
    try:
        _ = Job(workdir=myworkdir, stdout='stdout.txt', stderr='stderr.txt')
    except JobControllerException:
        jc_triggered = True

    assert jc_triggered, "Failed to raise exception if create job with no app"

    # Now with no workdir specified
    dummyapp = jobctrl.gen_default_app
    job1 = Job(app=dummyapp, stdout='stdout.txt', stderr='stderr.txt')
    wd_exist = job1.workdir_exists()
    assert not wd_exist  # , "No workdir specified, yet workdir_exists does not return False"
    stdout_exist = job1.stdout_exists()
    assert not stdout_exist
    f_exist = job1.file_exists_in_workdir('running_output.txt')
    assert not f_exist

    # Create job properly specified
    job2 = Job(app=dummyapp, workdir=myworkdir, stdout='stdout.txt', stderr='stderr.txt')

    # Workdir does exist
    wd_exist = job2.workdir_exists()
    assert wd_exist

    # Files do not exist
    stdout_exist = job2.stdout_exists()
    assert not stdout_exist
    stderr_exist = job2.stderr_exists()
    assert not stderr_exist
    f_exist = job2.file_exists_in_workdir('running_output.txt')
    assert not f_exist

    valerr_triggered = False
    try:
        job2.read_stdout()
    except ValueError:
        valerr_triggered = True
    assert valerr_triggered

    valerr_triggered = False
    try:
        job2.read_file_in_workdir('running_output.txt')
    except ValueError:
        valerr_triggered = True
    assert valerr_triggered

    # Now create files and check positive results
    with open("stdout.txt", "w") as f:
        f.write('This is stdout')
    with open("stderr.txt", "w") as f:
        f.write('This is stderr')
    with open("running_output.txt", "w") as f:
        f.write('This is running output')

    # job2 = Job(app = dummyapp, workdir = myworkdir, stdout = 'stdout.txt')
    # wd_exist = job2.workdir_exists()
    # assert wd_exist
    stdout_exist = job2.stdout_exists()
    assert stdout_exist
    stderr_exist = job2.stderr_exists()
    assert stderr_exist
    f_exist = job2.file_exists_in_workdir('running_output.txt')
    assert f_exist
    assert 'This is stdout' in job2.read_stdout()
    assert 'This is stderr' in job2.read_stderr()
    assert 'This is running output' in job2.read_file_in_workdir('running_output.txt')

    # Check if workdir does not exist
    job2.workdir = job2.workdir + '/bubbles'
    wd_exist = job2.workdir_exists()
    assert not wd_exist

    # Check timing
    assert not job2.launch_time and not job2.runtime and not job2.total_time
    job2.calc_job_timing()
    assert not job2.launch_time and not job2.runtime and not job2.total_time
    job2.launch_time = time.time()
    job2.calc_job_timing()
    assert job2.runtime is not None and job2.runtime == job2.total_time
    save_runtime, save_total_time = job2.runtime, job2.total_time
    job2.calc_job_timing()
    assert save_runtime == job2.runtime
    assert save_total_time == job2.total_time

    # Clean up
    os.chdir('../')
    shutil.rmtree(dirname)


if __name__ == "__main__":
    test_job_funcs()
