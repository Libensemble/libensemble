MPI Executor
==================

.. automodule:: mpi_executor
  :no-undoc-members:

.. autoclass:: MPI_Executor
  :show-inheritance:
  :inherited-members:

  .. automethod:: __init__

..  :member-order: bysource
..  :members: __init__, register_calc, submit, manager_poll

Class-specific attributes
-------------------------

Class-specific attributes can be set directly to alter the behavior of the MPI
executor. However, they should be used with caution, because they may not
be implemented in other executors.

:max_submit_attempts: (int) Maximum number of submission attempts for a given task. *Default: 5*.
:fail_time: (int) *Only if wait_on_run is set.* Maximum run time to failure in seconds that results in relaunch. *Default: 2*.

Example. To increase resilience against submission failures::

    taskctrl = MPI_Executor()
    taskctrl.max_launch_attempts = 10
    taskctrl.fail_time = 5

Note that the retry delay on launches starts at 5 seconds and increments by
5 seconds for each retry. So the 4th retry will wait for 20 seconds before
relaunching.
