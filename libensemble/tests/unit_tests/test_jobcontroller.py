#!/usr/bin/env python
#Integration Test of job controller module for libensemble
#Test does not require running full libensemble
import os
from libensemble.register import Register
from libensemble.controller import JobController, BalsamJobController

NCORES = 1

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
    sim_app = './my_simjob.x'
    if not os.path.isfile(sim_app):
        build_simfunc()

    registry = Register()
    jobctrl = JobController(registry = registry, auto_resources = False)
    registry.register_calc(full_path=sim_app, calc_type='sim')
    
# -----------------------------------------------------------------------------
# The following would typically be in the user sim_func
def polling_loop(jobctl, job, timeout_sec=6.0, delay=1.0):
    import time
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
    
    
def polling_loop_multijob(jobctl, job_list, timeout_sec=8.0, delay=0.5):
    import time
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

# Tests
def test_launch_and_poll():
    """ Test of launching and polling job and exiting on job finish"""
    setup_job_controller()
    jobctl = JobController.controller
    cores = NCORES
    args_for_sim = 'sleep 3'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)

def test_kill_on_file():
    """ Test of killing job based on something in output file"""    
    setup_job_controller()
    jobctl = JobController.controller    
    cores = NCORES
    args_for_sim = 'sleep 3 Error'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'USER_KILLED', "job.state should be USER_KILLED. Returned " + str(job.state)

def test_kill_on_timeout():
    setup_job_controller()
    jobctl = JobController.controller    
    cores = NCORES
    args_for_sim = 'sleep 12'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim)
    job = polling_loop(jobctl, job)
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'USER_KILLED', "job.state should be USER_KILLED. Returned " + str(job.state)

def test_launch_and_poll_multijobs():
    setup_job_controller()
    jobctl = JobController.controller  
    job_list = []
    cores = NCORES
    
    for j in range(3):
        #outfilename = 'out_' + str(j) + '.txt' #Could allow launch to generate outfile names based on job.id
        outfile = 'multijob_job_' + str(j) + '.out'
        sleeptime = 3 + j #Change args
        args_for_sim = 'sleep' + ' ' + str(sleeptime)
        rundir = 'run_' + str(sleeptime)
        job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim, stdout=outfile)
        job_list.append(job) 
        
    job_list_return = polling_loop_multijob(jobctl, job_list)
    for job in job_list_return:
        assert job.finished, "job.finished should be True. Returned " + str(job.finished)
        assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)
        

if __name__ == "__main__":
    test_launch_and_poll()    
    test_kill_on_file()
    test_kill_on_timeout()
    test_launch_and_poll_multijobs()
