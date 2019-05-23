Job Controller Module
=====================

.. automodule:: controller
  :no-undoc-members:
 
See  :doc:`example<overview>` for usage.

See the controller APIs for optional arguments. 

.. toctree::
   :maxdepth: 1
   :caption: Job Controllers:
   
   mpi_controller
   balsam_controller

Job Class
---------

Jobs are created and returned though the job_controller launch function. Jobs can be polled and
killed with the respective poll and kill functions. Job information can be queried through the job attributes
below and the query functions. Note that the job attributes are only updated when they are
polled/killed (or through other job or job controller functions).

.. autoclass:: Job
  :members:
  :exclude-members: calc_job_timing,check_poll
..  :member-order: bysource
..  :members: poll, kill, workdir_exists, file_exists_in_workdir, read_file_in_workdir, stdout_exists, read_stdout, stderr_exists, read_stderr


Job Attributes
--------------
                
Following is a list of job status and configuration attributes that can be retrieved from a job.

:NOTE: These should not be set directly. Jobs are launched by the job controller and job information can be queired through the job attributes below and the query functions. 

Job Status attributes include:

:job.state: (string) The job status. One of: ('UNKNOWN'|'CREATED'|'WAITING'|'RUNNING'|'FINISHED'|'USER_KILLED'|'FAILED')

:job.process: (process obj) The process object used by the underlying process manager (e.g., return value of subprocess.Popen)
:job.errcode: (int) The errorcode/return code used by the underlying process manager
:job.finished: (Boolean) True means job has finished running - not whether was successful
:job.success: (Boolean) Did job complete succesfully (e.g., returncode is zero)
:job.runtime: (int) Time in seconds that job has been running.
:job.total_time: (int) Total time from job submission to completion (only available when job is finished).

Run configuration attributes - Some will be auto-generated:

:job.workdir: (string) Work directory for the job
:job.name: (string) Name of job - auto-generated
:job.app: (app obj) Use application/executable, registered using jobctl.register_calc
:job.app_args: (string) Application arguments as a string  
:job.stdout: (string) Name of file where the standard output of the job is written (in job.workdir)
:job.stderr: (string) Name of file where the standard error of the job is written (in job.workdir)

A list of job_controller and job functions can be found under the Job Controller Module.

