Job Controller Overview
=======================

Many users' will wish to launch an application to the system from a :ref:`sim_f<api_sim_f>`
(or :ref:`gen_f<api_gen_f>`), running on a worker.

An MPI job, for example, could be initialized with a subprocess call to ``mpirun``, or
an alternative launcher such as ``aprun`` or ``jsrun``. The sim_f may then monitor this job,
check output, and possibly kill the job. The word ``job`` is used here to represent
a launch of an application to the system, where the system could be a supercomputer,
cluster, or any other provision of compute resources.

In order to remove the burden of system interaction from the user, and enable sim_f
scripts that are portable between systems, a job_controller interface is provided by
libEnsemble. The job_controller provides the key functions: ``launch()``, ``poll()`` and
``kill()``. libEnsemble auto-detects a number of system criteria, such as the MPI launcher, 
along with correct mechanisms for polling and killing jobs, on supported systems. It also
contains built in resilience, such as re-launching jobs that fail due to system factors.
User scripts that employ the job_controller interface will be portable between supported
systems. Job attributes can be queried to determine status after each poll. Functions are 
also provided to access and interrogate files in the job's working directory.

The Job Controller module can be used to submit
and manage jobs using a portable interface. Various back-end mechanisms may be
used to implement this interface on the system, including a proxy launcher and 
job management system, such as Balsam. Currently, these job_controllers launch
at the application level within an existing resource pool. However, submissions
to a batch schedular may be supported in the future.

At the top-level calling script, a job_controller is created and the executable
gen or sim applications are registered to it (these are applications that will
be runnable jobs). If an alternative job_controller, such as Balsam, is to be
used, then these can be created as in the example. Once in the user-side worker
code (sim/gen func), an MPI based job_controller can be retrieved without any
need to specify the type.

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

    import time

    # Will return controller (whether MPI or inherited such as Balsam).
    jobctl = MPIJobController.controller

    job = jobctl.launch(calc_type='sim', num_procs=8, app_args='input.txt',
                        stdout='out.txt', stderr='err.txt')

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

For a slightly more detailed working example see:

- libensemble/tests/regression_tests/test_jobcontroller_hworld.py

which uses sim function:

- libensemble/sim_funcs/job_control_hworld.py

For a more realistic example see:

- libensemble/tests/scaling_tests/forces/

which launches the forces.x application as an MPI job.

