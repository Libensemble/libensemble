from libensemble.controller import JobController
import numpy as np

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

def job_control_hworld(H, gen_info, sim_specs, libE_info):
    """ Test of launching and polling job and exiting on job finish"""
    jobctl = JobController.controller
    cores = sim_specs['cores']
    args_for_sim = 'sleep 3'
    job = jobctl.launch(calc_type='sim', num_procs=cores, app_args=args_for_sim, hyperthreads=True)
    jobstate = polling_loop(jobctl, job)
    
    assert job.finished, "job.finished should be True. Returned " + str(job.finished)
    assert job.state == 'FINISHED', "job.state should be FINISHED. Returned " + str(job.state)
    
    #This is temp - return something - so doing six_hump_camel_func again...
    batch = len(H['x'])
    O = np.zeros(batch,dtype=sim_specs['out'])
    for i,x in enumerate(H['x']):
        O['f'][i] = six_hump_camel_func(x)

    # v = np.random.uniform(0,10)
    # print('About to sleep for :' + str(v))
    # time.sleep(v)
    
    return O, gen_info
    
def six_hump_camel_func(x):
    """
    Definition of the six-hump camel
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2;
    term2 = x1*x2;
    term3 = (-4+4*x2**2) * x2**2;

    return  term1 + term2 + term3
