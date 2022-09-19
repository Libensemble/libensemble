MPI Executor - MPI apps
=======================

.. automodule:: mpi_executor
  :no-undoc-members:

.. autoclass:: MPIExecutor
  :show-inheritance:
  :inherited-members:
  :exclude-members: serial_setup, sim_default_app, gen_default_app, get_app, default_app, set_resources, get_task, set_workerID, set_worker_info, new_tasks_timing

  .. automethod:: __init__

..  :member-order: bysource
..  :members: __init__, register_app, submit, manager_poll

Class-specific Attributes
-------------------------

Class-specific attributes can be set directly to alter the behavior of the MPI
Executor. However, they should be used with caution, because they may not
be implemented in other executors.

:max_submit_attempts: (int) Maximum number of launch attempts for a given
                      task. *Default: 5*.
:fail_time: (int or float) *Only if wait_on_start is set.* Maximum run time to failure in
            seconds that results in relaunch. *Default: 2*.
:retry_delay_incr: (int or float) Delay increment between launch attempts in seconds.
            *Default: 5*. (E.g. First retry after 5 seconds, then 10 seconds, then 15, etc...)

Example. To increase resilience against submission failures::

    taskctrl = MPIExecutor()
    taskctrl.max_launch_attempts = 8
    taskctrl.fail_time = 5
    taskctrl.retry_delay_incr = 10

.. _customizer:
