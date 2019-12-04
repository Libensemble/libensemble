MPI Job Controller
==================

To create an MPI job controller, the calling script should contain::

    jobctl = MPIJobController()

See the controller API below for optional arguments.

.. automodule:: mpi_controller
  :no-undoc-members:

.. autoclass:: MPIJobController
  :show-inheritance:
  :inherited-members:

  .. automethod:: __init__

..  :member-order: bysource
..  :members: __init__, register_calc, launch, manager_poll

Class specific attributes
-------------------------

These attributes can be set directly to alter behaviour of the MPI job
controller. However, they should be used with caution, as they may not be
implemented in other job controllers.

:max_launch_attempts: (int) Maximum number of launch attempts for a given job. *Default: 5*.
:fail_time: (int) *Only if wait_on_run is set.* Maximum run-time to failure in seconds that results in re-launch. *Default: 2*.

Example. To increase resilience against launch failures::

    jobctrl = MPIJobController()
    jobctrl.max_launch_attempts = 10
    jobctrl.fail_time = 5

Note that an the re-try delay on launches starts at 5 seconds and increments by
5 seconds for each retry. So the 4th re-try will wait for 20 seconds before
re-launching.
