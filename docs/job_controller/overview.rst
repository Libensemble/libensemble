Job Controller Overview
=======================

The Job Controller module can be used by the worker or user-side code to issue
and manage jobs using a portable interface. Various back-end mechanisms may be
used to implement this interface on the system, either specified by the user at
the top-level, or auto-detected. The job_controller manages jobs using the
launch, poll and kill functions. Job attributes can then be queried to
determine status. Functions are also provided to access and interrogate files
in the job's working directory.

At the top-level calling script, a job_controller is created and the executable
gen or sim applications are registered to it (these are applications that will
be runnable jobs). If an alternative job_controller, such as Balsam, is to be
used, then these can be created as in the example. Once in the user-side worker
code (sim/gen func), an MPI based job_controller can be retrieved without any
need to specify the specific type.

**Example usage (code runnable with or without a Balsam backend):**

In calling function::

    sim_app = '/path/to/my/exe'
    USE_BALSAM = False
    
    if USE_BALSAM:
        from libensemble.balsam_controller import BalsamJobController
        jobctrl = BalsamJobController()    
    else:
        from libensemble.mpi_controller import MPIJobController
        jobctrl = MPIJobController()    
        
    jobctrl.register_calc(full_path=sim_app, calc_type='sim')
    
In user sim func::

    jobctl = MPIJobController.controller # This will work for inherited controllers also (e.g., Balsam)
    import time
    
    jobctl = MPIJobController.controller # Will return controller (whether Balsam or standard MPI).
    job = jobctl.launch(calc_type='sim', num_procs=8, app_args='input.txt', stdout='out.txt', stderr='err.txt') 
    
    while time.time() - start < timeout_sec:
        time.sleep(delay)
        
        # Has manager sent a finish signal
        jobctl.manager_poll()
        if jobctl.manager_signal == 'finish':
            job.kill()        
        
        # Poll job to see if completed
        job.poll()
        if job.finished:
            print(job.state)
            break
            
        # Check output file for error and kill job
        if job.stdout_exists():
            if 'Error' in job.read_stdout():
                job.kill()
                break

See the :doc:`job_controller<job_controller>` interface for API.  

For a more detailed working example see:

- libensemble/tests/regression_tests/test_jobcontroller_hworld.py

which uses sim function:

- libensemble/sim_funcs/job_control_hworld.py
