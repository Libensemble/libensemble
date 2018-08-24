#!/usr/bin/env python
#Integration Test of job controller module for libensemble
#Test does not require running full libensemble
import os
import sys
import time
import pytest
import socket
from libensemble.register import Register, BalsamRegister
from libensemble.controller import JobController, BalsamJobController

USE_BALSAM = False

NCORES = 1
sim_app = './my_simjob.x'


def setup_module(module):
    print ("setup_module      module:%s" % module.__name__)
    if JobController.controller is not None:
        ctrl = JobController.controller
        del ctrl
        JobController.controller = None
    if Register.default_registry:
        defreg = Register.default_registry
        del defreg
        Register.default_registry = None

def setup_function(function):
    print ("setup_function    function:%s" % function.__name__)
    if JobController.controller is not None:
        ctrl = JobController.controller
        del ctrl
        JobController.controller = None
    if Register.default_registry:
        defreg = Register.default_registry
        del defreg
        Register.default_registry = None

def teardown_module(module):
    print ("teardown_module   module:%s" % module.__name__)
    if JobController.controller is not None:
        ctrl = JobController.controller
        del ctrl
        JobController.controller = None
    if Register.default_registry:
        defreg = Register.default_registry
        del defreg
        Register.default_registry = None


#def setup_module(module):
    #import pdb; pdb.set_trace
    #if JobController.controller is not None:
        #print('controller found')

def build_simfunc():
    import subprocess

    #Build simfunc
    #buildstring='mpif90 -o my_simjob.x my_simjob.f90' # On cray need to use ftn
    buildstring='mpicc -o my_simjob.x simdir/my_simjob.c'
    #subprocess.run(buildstring.split(),check=True) #Python3.5+
    subprocess.check_call(buildstring.split())

# This would typically be in the user calling script
# Cannot test auto_resources here - as no workers set up.
def setup_job_controller():
    #sim_app = './my_simjob.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        registry = BalsamRegister()
        jobctrl = BalsamJobController(registry = registry, auto_resources = False)
    else:
        registry = Register()
        jobctrl = JobController(registry = registry, auto_resources = False)

    registry.register_calc(full_path=sim_app, calc_type='sim')

def setup_job_controller_noreg():
    #sim_app = './my_simjob.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        registry = BalsamRegister()
        jobctrl = BalsamJobController(auto_resources = False)
    else:
        registry = Register()
        jobctrl = JobController(auto_resources = False)

    registry.register_calc(full_path=sim_app, calc_type='sim')

def setup_job_controller_noapp():
    #sim_app = './my_simjob.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    if USE_BALSAM:
        registry = BalsamRegister()
        jobctrl = BalsamJobController(registry = registry, auto_resources = False)
    else:
        registry = Register()
        jobctrl = JobController(registry = registry, auto_resources = False)

# -----------------------------------------------------------------------------
# The following would typically be in the user sim_func
def polling_loop(jobctl, job, timeout_sec=0.5, delay=0.05):
    #import time
    start = time.time()

    while time.time() - start < timeout_sec:
        time.sleep(delay)
        print('Polling at time', time.time() - start)
        jobctl.poll(job)
        if job.finished: break
        elif job.state == 'WAITING': print('Job waiting to launch')
        elif job.state == 'RUNNING': print('Job still running ....')

        #Check output file for error
        if job.stdout_exists():
            if 'Error' in job.read_stdout():
                print("Found (deliberate) Error in ouput file - cancelling job")
                jobctl.kill(job)
                time.sleep(delay) #Give time for kill
                break
    if not job.finished:
        assert job.state == 'RUNNING', "job.state expected to be RUNNING. Returned: " + str(job.state)
        print("Job timed out - killing")
        jobctl.kill(job)
        time.sleep(delay) #Give time for kill
    return job


def polling_loop_multijob(jobctl, job_list, timeout_sec=4.0, delay=0.05):
    #import time
    start = time.time()

    while time.time() - start < timeout_sec:

        #Test all done - (return list of not-finished jobs and test if empty)
        active_list = [job for job in job_list if not job.finished]
        if not active_list:
            break

        for job in job_list:
            if not job.finished:
                time.sleep(delay)
                print('Polling job %d at time %f' % (job.id, time.time() - start))
                #job.poll()
                jobctl.poll(job)
                if job.finished: continue
                elif job.state == 'WAITING': print('Job %d waiting to launch' % (job.id))
                elif job.state == 'RUNNING': print('Job %d still running ....' % (job.id))

                #Check output file for error
                if job.stdout_exists():
                    if 'Error' in job.read_stdout():
                        print("Found (deliberate) Error in ouput file - cancelling job %d" % (job.id))
                        jobctl.kill(job)
                        time.sleep(delay) #Give time for kill
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
        #outfilename = 'out_' + str(j) + '.txt' #Could allow launch to generate outfile names based on job.id
        outfile = 'multijob_job_' + str(j) + '.out'
        sleeptime = 0.3 + (j*0.2) #Change args
        args_for_sim = 'sleep' + ' ' + str(sleeptime)
        rundir = 'run_' + str(sleeptime)
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

    #Try with no jobs set up
    A = jobctl.get_job('a')
    assert A is None, 'Job found when joblist should be empty'

    #Set up job and getid
    cores = NCORES
    args_for_sim = 'sleep 0'
    job0 = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    jobid = job0.id
    print("jobid is: {}".format(jobid))
    A = jobctl.get_job(jobid)
    assert A is job0 , 'Job get_job returned unexpected job' + str(A)
    job0 = polling_loop(jobctl, job0)

    #Get non-existent jobid
    A = jobctl.get_job(jobid+1)
    assert A is None, 'Job found when supplied jobid should not exist'


@pytest.mark.timeout(30)
def test_procs_and_machinefile_logic():
    """ Test of supplying various input configurations when auto_resources is False."""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))

    #Note: Could test job_partition routine directly - without launching jobs...

    # Testing machinefile
    setup_job_controller()
    jobctl = JobController.controller
    args_for_sim = 'sleep 0'

    machinefilename = "my_machinefile"
    cores = NCORES
    with open(machinefilename,'w') as f:
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
    except:
        assert 1
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
    except:
        assert 1
    else:
        assert 0

    # Testing no num_nodes (shouldn't fail)
    job = jobctl.launch(calc_type='sim',num_procs=2,ranks_per_node=2, app_args=args_for_sim)
    assert 1
    job = polling_loop(jobctl, job, delay=0.02)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    # Testing no ranks_per_node (shouldn't fail)
    job = jobctl.launch(calc_type='sim',num_nodes=1,num_procs=2, app_args=args_for_sim)
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
    jobctl.poll(job)
    #jobctl.set_kill_mode(wait_and_kill=True, wait_time=5)

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
        jobctl.poll(job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)
    jobctl.kill(job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)
    #Try polling after finish - should return with no effect
    jobctl.poll(job)
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

    #Try launching as gen when not registered as gen
    try:
        job = jobctl.launch(calc_type='gen', num_procs=cores, app_args=args_for_sim)
    except:
        assert 1
    else:
        assert 0

    registry = Register.default_registry
    registry.register_calc(full_path=sim_app, calc_type='gen')
    job = jobctl.launch(calc_type='gen', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

    #Try launching as 'alloc' which is not a type
    try:
        job = jobctl.launch(calc_type='alloc', num_procs=cores, app_args=args_for_sim)
    except:
        assert 1
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


def test_create_jobcontroller_no_registry():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    #import pdb;pdb.set_trace()
    try:
        jobctrl = JobController(auto_resources = False)
    except:
        assert 1
    else:
        assert 0


def test_launch_no_app():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller_noapp()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 0.1'
    try:
        job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    except:
        assert 1
    else:
        assert 0


def test_kill_job_with_no_launch():
    from libensemble.controller import Job
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES

    #Try kill invalid job
    try:
        jobctl.kill('myjob')
    except:
        assert 1
    else:
        assert 0

    # Create a job directly with no launch (Not supported for users)
    registry = Register.default_registry
    myapp = registry.sim_default_app
    job1 = Job(app = myapp, stdout = 'stdout.txt')
    try:
        jobctl.kill(job1)
    except:
        assert 1
    else:
        assert 0


def test_poll_job_with_no_launch():
    from libensemble.controller import Job
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES

    #Try poll invalid job
    try:
        jobctl.poll('myjob')
    except:
        assert 1
    else:
        assert 0

    # Create a job directly with no launch (Not supported for users)
    registry = Register.default_registry
    myapp = registry.sim_default_app
    job1 = Job(app = myapp, stdout = 'stdout.txt')
    try:
        jobctl.poll(job1)
    except:
        assert 1
    else:
        assert 0


def test_set_kill_mode():
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES

    signal_b4 = jobctl.kill_signal
    wait_and_kill_b4 = jobctl.wait_and_kill
    wait_time_b4 = jobctl.wait_time

    # Change nothing.
    jobctl.set_kill_mode()
    assert jobctl.kill_signal == signal_b4
    assert jobctl.wait_and_kill == wait_and_kill_b4
    assert jobctl.wait_time == wait_time_b4

    # While these options are set - wait_time will not be used. Result is warning.
    jobctl.set_kill_mode(signal='SIGKILL', wait_and_kill=False, wait_time=10)
    assert jobctl.kill_signal == 'SIGKILL'
    assert not jobctl.wait_and_kill
    assert jobctl.wait_time == 10

    # Now correct
    jobctl.set_kill_mode(signal='SIGTERM', wait_and_kill=True, wait_time=20)
    assert jobctl.kill_signal == 'SIGTERM'
    assert jobctl.wait_and_kill
    assert jobctl.wait_time == 20

    #Todo:
    #Testing wait_and_kill is harder - need to create a process that does not respond to sigterm in time.

    # Try set to unknown signal
    try:
        jobctl.set_kill_mode(signal='SIGDIE')
    except:
        assert 1
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
    #setup_module(__file__)
    test_launch_and_poll()
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
    setup_function(test_create_jobcontroller_no_registry)
    test_create_jobcontroller_no_registry()
    test_launch_no_app()
    test_kill_job_with_no_launch()
    test_poll_job_with_no_launch()
    test_set_kill_mode()
    test_job_failure()
    #teardown_module(__file__)

