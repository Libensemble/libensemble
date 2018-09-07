Job Controller Overview
=======================

The Job Controller module can be used by the worker or user-side code to issue and manage jobs using a portable interface. Various back-end mechanisms may be used to implement this interface on the system, either specified by the user at the top-level, or auto-detected. The job_controller manages jobs using the launch, poll and kill functions. Job attributes can then be queried to determine status. Functions are also provided to access and interrogate files in the job's working directory.

At the top-level calling script, a registry and job_controller are created and the executable gen or sim applications are registered to these (these are applications that will be runnable parallel jobs). If an alternative job_controller, such as Balsam, is to be used, then these can be created as in the example. Once in the user-side worker code (sim/gen func), the job_controller can be retrieved without any need to specify the type.

**Example usage (code runnable with or without a Balsam backend):**

In calling function::

    from libensemble.register import Register, BalsamRegister
    from libensemble.controller import JobController, BalsamJobController  
    sim_app = '/path/to/my/exe'
    USE_BALSAM = False
    
    if USE_BALSAM:
        registry = BalsamRegister()
        jobctrl = BalsamJobController(registry = registry)    
    else:
        registry = Register()
        jobctrl = JobController(registry = registry)    
        
    registry.register_calc(full_path=sim_app, calc_type='sim')
    
In user sim func::

    from libensemble.controller import JobController
    import time
    
    jobctl = JobController.controller #Will return controller (whether Balsam or standard).
    job = jobctl.launch(calc_type='sim', num_procs=8, app_args='input.txt', stdout='out.txt') 
    
    while time.time() - start < timeout_sec:
        time.sleep(delay)
        
        # Has manager sent a finish signal
        jobctl.manager_poll()
        if jobctl.manager_signal == 'finish':
            jobctl.kill(job)        
        
        # Poll job to see if completed
        jobctl.poll(job)
        if job.finished:
            print(job.state)
            break
            
        # Check output file for error and kill job
        if job.stdout_exists():
            if 'Error' in job.read_stdout():
                jobctl.kill(job)
                break

See the :doc:`job_controller<job_controller>` interface for API.  

For a more detailed working example see:

- libensemble/tests/regression_tests/test_jobcontroller_hworld.py

which uses sim function:

- libensemble/sim_funcs/job_control_hworld.py
