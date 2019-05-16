# !/usr/bin/env python
# Integration Test of job controller module for libensemble
# Test does not require running full libensemble
import os
import sys
import time
import pytest
import socket
from libensemble.resources import ResourcesException
from libensemble.controller import JobController, JobControllerException
from libensemble.controller import NOT_STARTED_STATES


USE_BALSAM = False

NCORES = 1
sim_app = './my_simjob.x'


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


def build_simfunc():
    import subprocess

    # Build simfunc
    # buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring = 'mpicc -o my_simjob.x simdir/my_simjob.c'
    # subprocess.run(buildstring.split(), check=True) # Python3.5+
    subprocess.check_call(buildstring.split())


# This would typically be in the user calling script
# Cannot test auto_resources here - as no workers set up.
def setup_job_controller():
    # sim_app = './my_simjob.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        from libensemble.balsam_controller import BalsamJobController
        jobctrl = BalsamJobController(auto_resources=False)
    else:
        from libensemble.mpi_controller import MPIJobController
        jobctrl = MPIJobController(auto_resources=False)

    jobctrl.register_calc(full_path=sim_app, calc_type='sim')


def setup_job_controller_noreg():
    # sim_app = './my_simjob.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        from libensemble.balsam_controller import BalsamJobController
        jobctrl = BalsamJobController(auto_resources=False)
    else:
        from libensemble.mpi_controller import MPIJobController
        jobctrl = MPIJobController(auto_resources=False)

    jobctrl.register_calc(full_path=sim_app, calc_type='sim')


def setup_job_controller_noapp():
    # sim_app = './my_simjob.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        from libensemble.balsam_controller import BalsamJobController
        jobctrl = BalsamJobController(auto_resources=False)
    else:
        from libensemble.mpi_controller import MPIJobController
        jobctrl = MPIJobController(auto_resources=False)
        if jobctrl.workerID is not None:
            sys.exit("Something went wrong in creating JobController")


# -----------------------------------------------------------------------------
# The following would typically be in the user sim_func
def polling_loop(jobctl, job, timeout_sec=0.5, delay=0.05):
    # import time
    start = time.time()

    while time.time() - start < timeout_sec:
        time.sleep(delay)
        print('Polling at time', time.time() - start)
        job.poll()
        if job.finished:
            break
        elif job.state == 'WAITING':
            print('Job waiting to launch')
        elif job.state == 'RUNNING':
            print('Job still running ....')

        # Check output file for error
        if job.stdout_exists():
            if 'Error' in job.read_stdout():
                print("Found(deliberate) Error in ouput file - cancelling job")
                jobctl.kill(job)
                time.sleep(delay)  # Give time for kill
                break
    if not job.finished:
        assert job.state == 'RUNNING', "job.state expected to be RUNNING. Returned: " + str(job.state)
        print("Job timed out - killing")
        jobctl.kill(job)
        time.sleep(delay)  # Give time for kill
    return job


def polling_loop_multijob(jobctl, job_list, timeout_sec=4.0, delay=0.05):
    # import time
    start = time.time()

    while time.time() - start < timeout_sec:

        # Test all done - (return list of not-finished jobs and test if empty)
        active_list = [job for job in job_list if not job.finished]
        if not active_list:
            break

        for job in job_list:
            if not job.finished:
                time.sleep(delay)
                print('Polling job %d at time %f' % (job.id, time.time() - start))
                job.poll()
                if job.finished:
                    continue
                elif job.state == 'WAITING':
                    print('Job %d waiting to launch' % (job.id))
                elif job.state == 'RUNNING':
                    print('Job %d still running ....' % (job.id))

                # Check output file for error
                if job.stdout_exists():
                    if 'Error' in job.read_stdout():
                        print("Found (deliberate) Error in ouput file - cancelling job %d" % (job.id))
                        jobctl.kill(job)
                        time.sleep(delay)  # Give time for kill
                        continue
    return job_list


# Tests ========================================================================================
def test_launch_and_poll():
    """ Test of launching and polling job and exiting on job finish"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.2'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)


def test_launch_wait_on_run():
    """ Test of launching job with wait_on_run """
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.2'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim, wait_on_run=True)
    assert job.state not in NOT_STARTED_STATES, "Job should not be in a NOT_STARTED state. State: " + str(job.state)
    jobctl.poll(job)
    if not job.finished:
        job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)


def test_kill_on_file():
    """ Test of killing job based on something in output file"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.1 Error'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'USER_KILLED', "job.state should be USER_KILLED. Returned " + str(job.state)


def test_kill_on_timeout():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 10'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'USER_KILLED', "job.state should be USER_KILLED. Returned " + str(job.state)


def test_launch_and_poll_multijobs():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    job_list = []
    cores = NCORES

    for j in range(3):
        # outfilename = 'out_' + str(j) + '.txt' # Could allow launch to generate outfile names based on job.id
        outfile = 'multijob_job_' + str(j) + '.out'
        sleeptime = 0.3 + (j*0.2)  # Change args
        args_for_sim = 'sleep' + ' ' + str(sleeptime)
        # rundir = 'run_' + str(sleeptime)
        job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim, stdout=outfile)
        job_list.append(job)

    job_list_return = polling_loop_multijob(jobctl, job_list)
    for job in job_list_return:
        assert job.finished, "job.finished should be True. Returned " + str(job.finished)
        assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)


def test_get_job():
    """Return job from given job id"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller

    # Try with no jobs set up
    A = jobctl.get_job('a')
    assert A is None, 'Job found when joblist should be empty'

    # Set up job and getid
    cores = NCORES
    args_for_sim = 'sleep 0'
    job0 = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    jobid = job0.id
    print("jobid is: {}".format(jobid))
    A = jobctl.get_job(jobid)
    assert A is job0, 'Job get_job returned unexpected job' + str(A)
    job0 = polling_loop(jobctl, job0)

    # Get non-existent jobid
    A = jobctl.get_job(jobid+1)
    assert A is None, 'Job found when supplied jobid should not exist'


@pytest.mark.timeout(30)
def test_procs_and_machinefile_logic():
    """ Test of supplying various input configurations when auto_resources is False."""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))

    # Note: Could test job_partition routine directly - without launching jobs...

    # Testing machinefile
    setup_job_controller()
    jobctl = JobController.controller
    args_for_sim = 'sleep 0'

    machinefilename = "my_machinefile"
    cores = NCORES
    with open(machinefilename, 'w') as f:
        for rank in range(cores):
            f.write(socket.gethostname() + '\n')

    job = jobctl.launch(calc_type='sim', machinefile=machinefilename, app_args=args_for_sim)
    job = polling_loop(jobctl, job, delay=0.02)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    # Testing num_procs = num_nodes*ranks_per_node (shouldn't fail)
    job = jobctl.launch(calc_type='sim', num_procs=6, num_nodes=2, ranks_per_node=3, app_args=args_for_sim)
    job = polling_loop(jobctl, job, delay=0.02)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    # Testing num_procs not num_nodes*ranks_per_node (should fail)
    try:
        job = jobctl.launch(calc_type='sim', num_procs=9, num_nodes=2, ranks_per_node=5, app_args=args_for_sim)
    except ResourcesException as e:
        assert e.args[0] == 'num_procs does not equal num_nodes*ranks_per_node'
    else:
        assert 0

    # Testing no num_procs (shouldn't fail)
    job = jobctl.launch(calc_type='sim', num_nodes=2, ranks_per_node=3, app_args=args_for_sim)
    assert 1
    job = polling_loop(jobctl, job, delay=0.02)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    # Testing nothing given (should fail)
    try:
        job = jobctl.launch(calc_type='sim', app_args=args_for_sim)
    except ResourcesException as e:
        assert e.args[0] == 'Need num_procs, num_nodes/ranks_per_node, or machinefile'
    else:
        assert 0

    # Testing no num_nodes (shouldn't fail)
    job = jobctl.launch(calc_type='sim', num_procs=2, ranks_per_node=2, app_args=args_for_sim)
    assert 1
    job = polling_loop(jobctl, job, delay=0.02)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    # Testing no ranks_per_node (shouldn't fail)
    job = jobctl.launch(calc_type='sim', num_nodes=1, num_procs=2, app_args=args_for_sim)
    assert 1
    job = polling_loop(jobctl, job, delay=0.02)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)


@pytest.mark.timeout(20)
def test_doublekill():
    """Test attempt to kill already killed job

    Kill should have no effect (except warning message) and should remain in state killed
    """
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 2.0'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job.poll()
    jobctl.wait_time = 5

    jobctl.kill(job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'USER_KILLED', "job.state should be USER_KILLED. Returned " + str(job.state)
    jobctl.kill(job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'USER_KILLED', "job.state should be USER_KILLED. Returned " + str(job.state)


@pytest.mark.timeout(20)
def test_finish_and_kill():
    """Test attempt to kill already finished job

    Kill should have no effect (except warning message) and should remain in state FINISHED
    """
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    while not job.finished:
        time.sleep(0.1)
        job.poll()
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)
    jobctl.kill(job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)
    # Try polling after finish - should return with no effect
    job.poll()
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)


@pytest.mark.timeout(20)
def test_launch_and_kill():
    """Test launching and immediately killing jobs with no poll"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 2.0'
    job_list = []
    jobctl.wait_time = 1
    for jobid in range(5):
        job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
        jobctl.kill(job)
        job_list.append(job)

    for job in job_list:
        assert job.finished, "job.finished should be True. Returned " + str(job.finished)
        assert job.state == 'USER_KILLED', "job.state should be USER_KILLED. Returned " + str(job.state)


def test_launch_as_gen():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.1'

    # Try launching as gen when not registered as gen
    try:
        job = jobctl.launch(calc_type='gen', num_procs=cores, app_args=args_for_sim)
    except JobControllerException as e:
        assert e.args[0] == 'Default gen app is not set'
    else:
        assert 0

    jobctl.register_calc(full_path=sim_app, calc_type='gen')
    job = jobctl.launch(calc_type='gen', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    # Try launching as 'alloc' which is not a type
    try:
        job = jobctl.launch(calc_type='alloc', num_procs=cores, app_args=args_for_sim)
    except JobControllerException as e:
        assert e.args[0] + e.args[1] == 'Unrecognized calculation type' + 'alloc'
    else:
        assert 0


def test_launch_default_reg():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller_noreg()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)


def test_launch_no_app():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller_noapp()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    try:
        _ = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    except JobControllerException as e:
        assert e.args[0] == 'Default sim app is not set'
    else:
        assert 0


def test_kill_job_with_no_launch():
    from libensemble.controller import Job
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller

    # Try kill invalid job
    try:
        jobctl.kill('myjob')
    except JobControllerException as e:
        assert e.args[0] == 'Invalid job has been provided'
    else:
        assert 0

    # Create a job directly with no launch (Not supported for users)
    myapp = jobctl.sim_default_app
    job1 = Job(app=myapp, stdout='stdout.txt')
    try:
        jobctl.kill(job1)
    except JobControllerException as e:
        assert e.args[0][:47] == 'Attempting to kill job job_my_simjob.x.simfunc_'
        assert e.args[0][49:] == ' that has no process ID - check jobs been launched'
    else:
        assert 0


def test_poll_job_with_no_launch():
    from libensemble.controller import Job
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller

    # Create a job directly with no launch (Not supported for users)
    myapp = jobctl.sim_default_app
    job1 = Job(app=myapp, stdout='stdout.txt')
    try:
        job1.poll()
    except JobControllerException as e:
        assert e.args[0][:35] == 'Polled job job_my_simjob.x.simfunc_'
        assert e.args[0][37:] == ' has no process ID - check jobs been launched'
    else:
        assert 0


def test_job_failure():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 1.0 Fail'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FAILED', "job.state should be FAILED. Returned " + str(job.state)


if __name__ == "__main__":
    # setup_module(__file__)
    test_launch_and_poll()
    test_launch_wait_on_run()
    test_kill_on_file()
    test_kill_on_timeout()
    test_launch_and_poll_multijobs()
    test_get_job()
    test_procs_and_machinefile_logic()
    test_doublekill()
    test_finish_and_kill()
    test_launch_and_kill()
    test_launch_as_gen()
    test_launch_default_reg()
    test_launch_no_app()
    test_kill_job_with_no_launch()
    test_poll_job_with_no_launch()
    test_job_failure()
    # teardown_module(__file__)
