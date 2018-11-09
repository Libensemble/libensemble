Balsam Job Controller
=====================

To create a Balsam job controller, the calling script should contain::

    jobctr = BalsamJobController()

The Balsam job controller inherits from the MPI job controller. See the
:doc:`MPIJobController<mpi_controller>` for shared API. Any differences are 
shown below.

.. automodule:: balsam_controller
  :no-undoc-members:
  
.. autoclass:: BalsamJobController
  :show-inheritance:
..  :inherited-members:
..  :member-order: bysource  
..  :members: __init__, launch, poll, manager_poll, kill, set_kill_mode

.. autoclass:: BalsamJob
  :show-inheritance:
  :member-order: bysource
..  :members: workdir_exists, file_exists_in_workdir, read_file_in_workdir, stdout_exists, read_stdout  
..  :inherited-members:
