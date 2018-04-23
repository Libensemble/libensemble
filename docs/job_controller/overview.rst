==============
Job Controller
==============

The Job Controller module can be used by the worker or user-side code to issue and manage jobs using a portable interface. Various back-end mechanisms may be used to implement this interface on the system, either specified by the user at the top-level, or auto-detected. The job_controller manages jobs using the launch, poll and kill functions. Job attributes can then be queried to determine status. Functions are also provided to access and interrogate files in the job's working directory.

At the top-level calling script, a registry and job_controller are created and the executable gen or sim applications are registered to these (these are applications that will be runnable parallel jobs). If an alternative job_controller, such as Balsam, is to be used, then these can be created as in the example. Once in the user-side worker code (sim/gen func), the job_controller can be retrieved without any need to specify the type.

**Example usage (code runnable with or without Balsam backend):**

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
        jobctl.poll(job)
        if job.finished:
            print(job.state)
            break
         if job.stdout_exists():
            if 'Error' in job.read_stdout():
                jobctl.kill(job)
                break
                
Following is a list of job status and configuration attributes that can be retrieved from job.

Job Status attributes include::

    job.state = ('UNKNOWN'|'CREATED'|'WAITING'|'RUNNING'|'FINISHED'|'USER_KILLED'|'FAILED')
    job.process = The process object used by the underlying process manager (e.g. return value of subprocess.Popen)
    job.errcode = The errorcode/return code used by the underlying process manager
    job.finished = True means job has finished running - not whether was successful
    job.success = Did job complete succesfully (e.g. returncode is zero)
        
Run configuration attributes - Some will be auto-generated::

    job.workdir = Work directory for the job
    job.app = Use application/executable, registered using registry.register_calc
    job.app_args = Application arguments as a string  
    job.num_procs = Total number of processors for job
    job.num_nodes = Number of nodes for job
    job.ranks_per_node = Ranks per node for job
    job.machinefile = Name of machinefile is provided
    job.stdout = Name of file where the job's standard output is written
    job.jobname = Name of job - auto-generated 
