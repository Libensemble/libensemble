Job Controller Overview
=======================

Users who wish to launch jobs to a system from a :ref:`sim_f<api_sim_f>` (or :ref:`gen_f<api_gen_f>`),
running on a worker have several options.

Typically, an MPI job could be initialized with a subprocess call to
``mpirun`` or an alternative launcher such as ``aprun`` or ``jsrun``. The ``sim_f``
may then monitor this job, check output, and possibly kill the job. We use "job"
to represent an application launch to the system, which may be a supercomputer,
cluster, or other provision of compute resources.

A **job_controller** interface is provided by libEnsemble to remove the burden of
system interaction from the user and ease writing portable user scripts that
launch applications. The job_controller provides the key functions: ``launch()``,
``poll()`` and ``kill()``. Job attributes can be queried to determine status after
each poll. To implement these functions, libEnsemble auto-detects system criteria
such as the MPI launcher and mechanisms to poll and kill jobs on supported systems.
libEnsemble's job_controller is resilient, and can re-launch jobs that fail due
to system factors.

Functions are also provided to access and interrogate files in the job's working directory.

Various back-end mechanisms may be used by the job_controller to best interact
with each system, including proxy launchers or job management systems like
Balsam_. Currently, these job_controllers launch at the application level within
an existing resource pool. However, submissions to a batch scheduler may be
supported in the future.

In a calling script, a job_controller object is created and the executable
generator or simulation applications are registered to it for launch. If an
alternative job_controller like Balsam will be used, then the applications can be
registered like in the example below. Once in the user-side worker code (sim/gen func),
an MPI based job_controller can be retrieved without any need to specify the type.

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

    timeout_sec = 600
    poll_delay_sec = 1

    while(not job.finished):

        # Has manager sent a finish signal
        jobctl.manager_poll()
        if jobctl.manager_signal == 'finish':
            job.kill()
            my_cleanup()

        # Check output file for error and kill job
        elif job.stdout_exists():
            if 'Error' in job.read_stdout():
                job.kill()

        elif job.runtime > timeout_sec:
            job.kill()  # Timeout

        else:
            time.sleep(poll_delay_sec)
            job.poll()

    print(job.state)  # state may be finished/failed/killed

See the :doc:`job_controller<job_controller>` interface for API.

For a more realistic example see:

- libensemble/tests/scaling_tests/forces/

which launches the forces.x application as an MPI job.

.. _Balsam: https://balsam.readthedocs.io/en/latest/
